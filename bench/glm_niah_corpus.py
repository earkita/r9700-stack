#!/usr/bin/env python3
"""Freeze exact-length GLM NIAH prompts for glm_quality.py (no inference).

Run with the checkpoint's local tokenizer. The serving runner independently
checks the resulting lengths through /tokenize before issuing any request.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    from transformers import AutoTokenizer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[8192, 32768])
    parser.add_argument("--depths", type=int, nargs="+", default=[5, 35, 65, 95])
    parser.add_argument("--records", type=int, default=2048,
                        help="Background records available; use 4096 for 128K")
    args = parser.parse_args()
    if args.records < 1:
        parser.error("--records must be positive")
    args.out.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    system = ("You are performing an exact long-context retrieval test. Find the one "
              "CRITICAL NEEDLE record in the archive. When asked, return only its secret code and no other text.")
    records = [f"Record {i:06d}. The archive team reviewed the ordinary maintenance report. "
               "The storage checks completed normally, the scheduled inspection found no changes, "
               "and the next review remains on the calendar. This is a routine background entry.\n"
               for i in range(args.records)]
    cases, manifest = [], []
    for length in args.lengths:
        for depth in args.depths:
            if not 0 < depth < 100:
                raise ValueError("Depth must be between 0 and 100")
            suffix = hashlib.sha256(f"glm-niah-v1:{length}:{depth}".encode()).hexdigest()[:6].upper()
            secret = f"R9700-NIAH-{depth:03d}-{suffix}"
            needle = f"CRITICAL NEEDLE RECORD. The secret code is {secret}. Preserve this code exactly.\n"

            def messages(count, padding=0):
                split = round(count * depth / 100)
                archive = "".join(records[:split]) + needle + "".join(records[split:count])
                user = ("BEGIN ARCHIVE\n" + archive + "Routine padding:" + " x" * padding
                        + "\nEND ARCHIVE\nWhat is the secret code in the CRITICAL NEEDLE record? Return only the code.")
                return [{"role": "system", "content": system}, {"role": "user", "content": user}]

            def render(msgs, tokenize=True):
                return tokenizer.apply_chat_template(msgs, tokenize=tokenize,
                            add_generation_prompt=True, reasoning_effort="high", return_dict=False)

            lo, hi = 0, len(records)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if len(render(messages(mid))) <= length:
                    lo = mid
                else:
                    hi = mid - 1
            gap = length - len(render(messages(lo)))
            if gap < 0:
                raise ValueError("Requested length is too short")
            # A boundary merge at the first padding word can add/remove a
            # token. Search a bounded neighbourhood and verify the final text.
            for padding in range(max(0, gap - 3), gap + 4):
                msgs = messages(lo, padding)
                tokens = render(msgs)
                if len(tokens) == length:
                    break
            else:
                raise ValueError(f"Could not construct exact token length {length}")
            rendered = render(msgs, tokenize=False)
            assert rendered.count(secret) == 1
            offset = len(tokenizer.encode(rendered[:rendered.index(needle)], add_special_tokens=False))
            case_id = f"niah_{length}_depth{depth:03d}"
            cases.append(dict(id=case_id, messages=msgs, expected=secret, scoring="exact",
                              task="niah", key_pattern=r"R9700-NIAH-[0-9]{3}-[0-9A-F]{6}"))
            manifest.append(dict(id=case_id, prompt_tokens=length, needle_token_offset=offset,
                                 actual_depth=offset/length, nominal_depth=depth/100,
                                 expected=secret, records=lo, padding_words=padding,
                                 token_ids_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest()))
            print(json.dumps(manifest[-1]), flush=True)
    corpus = json.dumps(cases, indent=2) + "\n"
    (args.out / "corpus.json").write_text(corpus)
    (args.out / "manifest.json").write_text(json.dumps({
        "design": "Single explicit needle in coherent numbered archive records; no inference during preparation",
        "thinking": "high", "tokenizer": args.model, "cases": manifest,
        "corpus_sha256": hashlib.sha256(corpus.encode()).hexdigest(),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
