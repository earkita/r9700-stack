# GLM-5.3-Flash conservative fallback

[Profile](glm-5.3-flash.sh) · [Shared configuration](README.md) · [Testing](testing.md)

Text-only TP8, one sequence, context 65536, eager execution, APC off, no drafter
and no weight offload. Stock model/quantization paths are the default. Image
`r9700/vllm:dev`; model defaults to `amd/GLM-5.3-Flash-Quark-MXFP4` rather than
inheriting the C2/C4 checkpoint path or image.

```bash
bash serve/baseline-earkita/glm/glm-5.3-flash.sh DRYRUN=1
bash serve/baseline-earkita/glm/glm-5.3-flash.sh
```

This independently invokes `serve/serve.sh`. The script documents optional W4A4
and graph overrides; they are experimental and are not baseline defaults.
The C4 and LMCache timing reports do not apply to this fallback configuration.
