#!/bin/bash
# GLM-5.3-Flash MXFP4 + DFlash K4, 8x R9700. Wspolny launcher: serve.sh.
# Nadpisania KEY=value jak w flashnext.sh. Podglad: DRYRUN=1.
# Sprawdzony wariant: MAXLEN=262144 NSEQ=2 KVMEM=4.125; wiekszy kontekst wymaga testu.

# Rozmiary grafow musza odpowiadac liczbie sekwencji; uwzglednij tez argumenty CLI.
NSEQ=${NSEQ:-2}
MAXLEN=${MAXLEN:-auto}
for arg in "$@"; do
  case "$arg" in
    NSEQ=*) NSEQ=${arg#*=} ;;
    MAXLEN=*) MAXLEN=${arg#*=} ;;
  esac
done
case "$NSEQ" in
  1) graphs=5 ;;
  2) graphs=5,10 ;;
  4) graphs=5,10,15,20 ;;
  *) echo 'NSEQ must be 1, 2 or 4.' >&2; exit 2 ;;
esac
[ "$MAXLEN" = auto ] && MAXLEN=-1
serve_dir=$(dirname "$(realpath "$0")")

exec env \
  REPO="${REPO:-$(dirname "$serve_dir")}" `# Katalog pluginu na hoscie.` \
  IMG="${IMG:-r9700/vllm:glm53-plugin-e97573215}" `# Obraz z przypieta wersja vLLM.` \
  MODELS_DIR="${MODELS_DIR:-/mnt/ai/models/glm}" `# Host -> /models w kontenerze.` \
  MODEL="${MODEL:-/models/GLM-5.3-Flash-Quark-MXFP4}" `# Model glowny, eksperci MXFP4.` \
  DRAFT="${DRAFT:-/models/GLM-5.3-Flash-DFlash2-HF-bf582e4}" `# Drafter, wagi BF16.` \
  NAME="${NAME:-glm53-flash-c2}" \
  SERVED_MODEL_NAME=glm-5.3-flash \
  HOST="${HOST:-0.0.0.0}" \
  PORT="${PORT:-8080}" `# Port API.` \
  DOCKER_SUDO="${DOCKER_SUDO-}" \
  DRYRUN="${DRYRUN:-0}" `# 1 = tylko pokaz komende.` \
  REPLACE=0 `# Nie usuwaj istniejacego kontenera.` \
  BUILD_KERNELS=0 `# Uzyj juz zbudowanej biblioteki kerneli.` \
  MODEL_PREFLIGHT=r9700_vllm.models.glm5_next \
  GPUS=0,1,2,3,4,5,6,7 \
  TP=8 `# Model rozlozony na 8 GPU; drafter tez TP8 (SPEC_EXTRA).` \
  MAXLEN="$MAXLEN" `# auto (-1): maks. JEDNEGO kontekstu; wejscie + wyjscie, takze thinking.` \
  NSEQ="$NSEQ" `# Domyslnie 2 aktywne zadania; auto nie gwarantuje 2 pelnych kontekstow naraz.` \
  NBT="${NBT:-2048}" `# Prefill i minimum cache encodera; 2048 pokrywa pelny limit obrazu.` \
  UTIL=0.90 `# Jawny KVMEM ma pierwszenstwo dla rozmiaru KV.` \
  KVMEM="${KVMEM:-4.125}" `# GiB KV na GPU, wspolne dla wszystkich zadan.` \
  KV_DTYPE=fp8_e4m3 `# Format KV modelu glownego.` \
  DRAFT_KV_DTYPE=fp8_e4m3 `# Format KV draftera; jego wagi pozostaja BF16.` \
  PREFIX_CACHE=1 `# APC: ponowne wykorzystanie wspolnych prefiksow.` \
  OFFLOAD_GB=0 `# Wagi pozostaja na GPU; offload KV tez nie jest skonfigurowany.` \
  MTP= `# Wbudowane MTP wylaczone; korzystamy z DFlash.` \
  EAGER=0 `# Uzywaj grafow, zamiast wymuszac eager.` \
  SPEC=4 `# K4: drafter proponuje 4 tokeny na krok.` \
  SPEC_METHOD=dflash \
  DRAFT_ATTN=TRITON_ATTN `# Backend attention draftera.` \
  `# Parametry draftera: zachowaj koncowy blok APC, TP8, przypieta rewizja wag.` \
  SPEC_EXTRA='"disable_eagle_block_drop":true,"draft_tensor_parallel_size":8,"revision":"bf582e4eacc1810f76656d1811693ff6c6737d2a"' \
  CGMODE=FULL_DECODE_ONLY `# Grafy tylko dla decode.` \
  CGSIZES="$graphs" `# C1: 5; C2: 5,10; C4: 5,10,15,20.` \
  REASONING_PARSER=glm45 `# Oddzielanie thinking od odpowiedzi.` \
  TOOL_CALL_PARSER=glm47 `# Parsowanie wywolan narzedzi.` \
  LMONLY= `# Tekst + obrazy; fallback glm-5.3-flash.sh pozostaje tekstowy.` \
  `# Do 100 obrazow w calej historii zadania, bez wideo; do 2048 tokenow na obraz.` \
  `# max_pixels obejmuje 2 powtorzone klatki: 2 * 28 * 28 * 2048; zgodny budzet profilowania.` \
  EXTRA='--disable-custom-all-reduce --mm-encoder-attn-backend FLASH_ATTN --limit-mm-per-prompt {"image":100,"video":0} --mm-processor-kwargs {"max_image_tokens":2048,"max_pixels":3211264}' \
  ATTN= CHAT_TEMPLATE= OVERLAYS= WRAP= \
  PROF=0 `# Profiler wylaczony.` \
  DOCKER_ARGS='-e FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE' `# Vision: Flash Attention Triton RDNA4; omija wadliwa sciezke Torch SDPA.` \
  P2P=1 `# Komunikacja GPU-GPU.` \
  HWQ=1 MWAITX=1 PIDNS=--pid=host \
  R9K_GLM_BASELINE=1 R9K_PLATFORM=1 VLLM_PLUGINS=r9700,r9700_glm \
  R9K_DISABLE=quant,models,mtp,gdn,attn `# Wylacz ogolne patche; aktywne sa adaptery GLM.` \
  VLLM_ROCM_USE_AITER=1 \
  R9K_GLM_FP8_SPARSE=1 `# Obsluga sparse attention z FP8 KV na RDNA4.` \
  R9K_GLM_DFLASH=1 `# Adapter GLM/DFlash.` \
  R9K_GLM_MOE=w4a4 `# Kernel ekspertow z wagami i aktywacjami MXFP4.` \
  R9K_GLM_INDEXER_GEMV=1 `# Zoptymalizowany GEMV indexera.` \
  R9K_GLM_INDEXER_WORKSPACE=bounded `# Bufor wg NSEQ i index_kpool=4.` \
  R9K_GLM_DFLASH_SHARD_FC=1 `# Podziel projekcje draftera na TP8.` \
  R9K_GLM_DFLASH_CACHE=shared `# Wspolna pula KV modelu i draftera.` \
  R9K_GLM_APC_BOUNDARY=fixed `# Poprawka granicy zapisu stanow APC.` \
  R9K_GLM_W4A4_MAX_ROWS=32 `# Limit wierszy dla szybkiej sciezki W4A4.` \
  R9K_GLM_W4A4_BATCH=packed `# Kernel dla wielu wierszy bez rozpakowania calych wag.` \
  R9K_ARN_MAX_KB=16 R9K_AR4=0 R9K_AR_QUANT=0 \
  R9K_EXPERT_CACHE_SLOTS=0 `# Bez cache ekspertow offloadowanych z CPU.` \
  R9K_TARGET_LMHEAD=stock R9K_DRAFT_LMHEAD=stock \
  "$@" \
  MAXLEN="$MAXLEN" \
  bash "$serve_dir/serve.sh"
