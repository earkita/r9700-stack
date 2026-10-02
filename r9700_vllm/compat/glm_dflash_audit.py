"""Opt-in, bounded eager capture of actual GLM auxiliary states.

Set R9K_GLM_DFLASH_AUDIT_DIR to an artifact directory; create its `capture`
file only after startup. Never enable this diagnostic during a benchmark.
Each GPU saves two real prefill/verify samples per tap, with eight rows each.
"""
import os
from pathlib import Path


def install_draft_capture():
    from functools import wraps
    from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model
    original = DFlashQwen3Model.__init__
    project = DFlashQwen3Model._project_context_kv

    def save(model, kind, data):
        import torch
        from vllm.distributed import get_tensor_model_parallel_rank
        path = Path(os.environ["R9K_GLM_DFLASH_AUDIT_DIR"])
        if not (path / "capture").exists():
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("DFlash state capture requires EAGER=1")
        phase = "verify" if data["original_rows"] <= 8 else "prefill"
        file = path / f"{kind}-r{get_tensor_model_parallel_rank()}-{phase}.pt"
        if file.exists():
            return
        torch.save({k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in data.items()}, file)

    @wraps(original)
    def init(self, *, vllm_config, start_layer_id=0, prefix=""):
        if not vllm_config.model_config.enforce_eager:
            raise RuntimeError("DFlash state audit requires EAGER=1")
        original(self, vllm_config=vllm_config, start_layer_id=start_layer_id, prefix=prefix)
        if self.use_aux_hidden_state:
            def hook(fc, inputs, output):
                # Rows are bounded before copying; weights are saved once/phase.
                result = output[0] if isinstance(output, tuple) else output
                save(self, "fc", dict(x=inputs[0][:8], y=result[:8], w=fc.weight,
                                      original_rows=len(inputs[0])))
            self.fc.register_forward_hook(hook)

    @wraps(project)
    def project_kv(self, context_states, num_ctx, num_layers, num_kv_heads, head_dim):
        k, v = project(self, context_states, num_ctx, num_layers, num_kv_heads, head_dim)
        save(self, "kv", dict(x=context_states[:8], k=k[:, :8], v=v[:, :8], original_rows=len(context_states),
                               w=self._fused_kv_weight, norm=self._hidden_norm_weight,
                               eps=self._rms_norm_eps, layers=num_layers, heads=num_kv_heads, dim=head_dim))
        return k, v

    DFlashQwen3Model.__init__ = init
    DFlashQwen3Model._project_context_kv = project_kv


def capture_aux(model, layer, state, aux):
    import torch
    from vllm.distributed import get_tensor_model_parallel_rank
    path = Path(os.environ["R9K_GLM_DFLASH_AUDIT_DIR"])
    if not (path / "capture").exists():
        return
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("DFlash state capture requires EAGER=1")
    hidden, residual, post, comb = state
    phase = "verify" if hidden.shape[0] <= 8 else "prefill"
    tap = model.aux_hidden_state_layers[len(model._r9k_aux)]
    counts = getattr(model, "_r9k_audit_counts", {})
    key = (tap, phase)
    count = counts.get(key, 0)
    if count >= 2:
        return
    indices = torch.linspace(0, hidden.shape[0] - 1, min(8, hidden.shape[0]), device=hidden.device).long()
    tensors = {name: None if value is None else value[indices].detach().cpu()
               for name, value in zip(("hidden", "residual", "post", "comb", "aux"), (*state, aux))}
    rank = get_tensor_model_parallel_rank()
    torch.save(dict(tap=tap, phase=phase, rank=rank, original_rows=len(hidden),
                    indices=indices.cpu(), **tensors), path / f"aux-r{rank}-tap{tap}-{phase}-{count}.pt")
    counts[key] = count + 1
    model._r9k_audit_counts = counts


def main():
    import argparse
    import json
    import torch
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("--draft-checkpoint", type=Path, required=True, help="Local pinned DFlash model.safetensors")
    args = p.parse_args()
    torch.set_num_threads(8)
    from safetensors import safe_open
    with safe_open(args.draft_checkpoint, framework="pt", device="cpu") as f:
        checkpoint_fc = f.get_tensor("fc.weight")
        checkpoint_norm = f.get_tensor("hidden_norm.weight")
        checkpoint_kv = [(f.get_tensor(f"layers.{i}.self_attn.k_proj.weight"),
                          f.get_tensor(f"layers.{i}.self_attn.v_proj.weight")) for i in range(5)]
    results = []
    for file in sorted(args.directory.glob("aux-*.pt")):
        data = torch.load(file, map_location="cpu", weights_only=True)
        x, residual, post, comb, actual = (data[k] for k in ("hidden", "residual", "post", "comb", "aux"))
        ref = x if post is None else (
            torch.einsum("sij,sih->sjh", comb.double(), residual.double())
            + post.double() * x.double().unsqueeze(1)).to(x.dtype).mean(1)
        error = (actual.float() - ref.float()).abs()
        # BF16 mHC materialization and averaging: allow two BF16 ulps at
        # the output scale, with a small floor for cancellation near zero.
        tolerance = ref.float().abs() * .016 + .015625
        passed = bool(torch.isfinite(actual).all() and (error <= tolerance).all())
        results.append(dict(file=file.name, tap=data["tap"], phase=data["phase"], rank=data["rank"],
                            max_abs=float(error.max()), mean_abs=float(error.mean()), passed=passed))
    coverage = {(r["rank"], r["tap"], r["phase"]) for r in results}
    expected = {(r, t, p) for r in range(8) for t in (6, 15, 25, 34, 43) for p in ("prefill", "verify")}
    projections = []
    for phase in ("prefill", "verify"):
        files = [args.directory / f"fc-r{rank}-{phase}.pt" for rank in range(8)]
        if not all(f.exists() for f in files):
            continue
        data = [torch.load(f, weights_only=True) for f in files]
        x, actual = data[0]["x"], data[0]["y"]
        sharded = data[0]["w"].shape[1] == 2560
        w = torch.cat([d["w"] for d in data], 1) if sharded else data[0]["w"]
        weight_match = torch.equal(w, checkpoint_fc)
        ref = (x.double() @ w.double().T).float()
        err = (actual.float() - ref).abs()
        # TP reduction rounds partial BF16 products; use an L2 error bound,
        # avoiding a relative-per-element bound near cancellation to zero.
        relative_l2 = float(err.norm() / ref.norm().clamp_min(1e-12))
        same_input = all(torch.equal(x, d["x"]) for d in data)
        same_output = all(torch.equal(actual, d["y"]) for d in data)
        projections.append(dict(kind="fc", phase=phase, sharded=sharded,
                                relative_l2=relative_l2, max_abs=float(err.max()),
                                weight_match=weight_match,
                                passed=weight_match and same_input and same_output and relative_l2 < .01))
    for file in sorted(args.directory.glob("kv-*.pt")):
        d = torch.load(file, weights_only=True)
        rank = int(file.name.split("-r")[1].split("-")[0])
        expected_weight = torch.cat([w[rank * 128:(rank + 1) * 128] for pair in checkpoint_kv for w in pair])
        weight_match = torch.equal(d["w"], expected_weight) and torch.equal(d["norm"], checkpoint_norm)
        x = d["x"].float()
        normed = ((x * torch.rsqrt(x.square().mean(-1, keepdim=True) + d["eps"]))
                  * d["norm"].float()).bfloat16()
        ref = normed.double() @ d["w"].double().T
        ref = ref.reshape(len(x), d["layers"], 2, d["heads"], d["dim"]).permute(2, 1, 0, 3, 4)
        actual = torch.stack([d["k"], d["v"]]).double()
        err = (actual - ref).abs()
        relative_l2 = float(err.norm() / ref.norm().clamp_min(1e-12))
        projections.append(dict(kind="kv", file=file.name, relative_l2=relative_l2,
                                max_abs=float(err.max()), weight_match=weight_match,
                                passed=weight_match and relative_l2 < .01))
    projection_coverage = len(projections) == 18
    report = dict(samples=results, projections=projections,
                  complete=coverage == expected and projection_coverage,
                  passed=bool(results) and coverage == expected and projection_coverage
                         and all(r["passed"] for r in results + projections))
    (args.directory / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(samples=len(results), complete=report["complete"], passed=report["passed"],
                          max_abs=max((r["max_abs"] for r in results), default=None))))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
