# GLM-5.3-Flash C2

[Profile](glm-5.3-flash-c2.sh) · [Shared configuration](README.md) · [Testing](testing.md)

Two active sequences on TP8, DFlash K4, FP8 E4M3 target/draft KV, APC in VRAM,
4.125 GiB KV per GPU, NBT2048 and decode graphs 5/10. Text and images are enabled.
`MAXLEN=auto` selects the runtime capacity for one request; it does not reserve
two complete contexts. The previously checked fixed limit is 262144 tokens.

```bash
bash serve/baseline-earkita/glm/glm-5.3-flash-c2.sh DRYRUN=1
bash serve/baseline-earkita/glm/glm-5.3-flash-c2.sh
```

This independently invokes `serve/serve.sh`. It does not launch LMCache by default.
Use the [combined C4+LMCache profile](glm-5.3-flash-c4-lmcache.md) for the qualified
RAM/disk setup. Its C2-client measurements were made on an NSEQ=4 server and are
not measurements of this NSEQ=2 profile. See the shared reference for vision
qualification and capacity limits; C2 vision was checked by dry-run/CPU tests.
