#!/bin/bash
# Eksperyment: GLM-5.3-Flash MXFP4 + wbudowane MTP, 8x R9700.
# Notatki i stan walidacji: glm-5.3-flash-c4-mtp.md obok tego skryptu.
# Nadpisania KEY=value jak w flashnext.sh. Podglad: DRYRUN=1.
# Profil nie jest zakwalifikowany; nie zastępuje C4/DFlash ani DeepSeeka.

# Rozmiary grafow musza odpowiadac liczbie sekwencji; uwzglednij tez argumenty CLI.
NSEQ=${NSEQ:-4}
MAXLEN=${MAXLEN:-auto}
MTP=${MTP:-1} # Liczba tokenow spekulacyjnych; jedna warstwa wag jest powtarzana.
for arg in "$@"; do
  case "$arg" in
    NSEQ=*) NSEQ=${arg#*=} ;;
    MAXLEN=*) MAXLEN=${arg#*=} ;;
    MTP=*) MTP=${arg#*=} ;;
    DRAFT=*|SPEC=*|SPEC_METHOD=*|KV_OFFLOAD=*|KV_TRANSFER_CONFIG=*)
      echo 'This profile uses built-in MTP only; DFlash and external KV offload are not supported.' >&2
      exit 2 ;;
  esac
done
case "$MTP" in
  1|2|3|5) ;;
  *) echo 'MTP must be 1, 2, 3 or 5.' >&2; exit 2 ;;
esac
case "$NSEQ" in
  1|2|4) ;;
  *) echo 'NSEQ must be 1, 2 or 4.' >&2; exit 2 ;;
esac
graphs=
for ((clients=1; clients<=NSEQ; clients++)); do
  size=$((clients * (MTP + 1)))
  graphs+="${graphs:+,}$size"
done
[ "$MAXLEN" = auto ] && MAXLEN=-1
serve_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

exec env \
  REPO="${REPO:-$(dirname "$serve_dir")}" `# Katalog pluginu na hoscie.` \
  IMG="${IMG:-r9700/vllm:glm53-plugin-e97573215}" `# Obraz z przypieta wersja vLLM.` \
  MODELS_DIR="${MODELS_DIR:-/mnt/ai/models/glm}" `# Host -> /models w kontenerze.` \
  MODEL="${MODEL:-/models/GLM-5.3-Flash-Quark-MXFP4}" `# Model glowny, eksperci MXFP4.` \
  DRAFT= `# Wylacz osobny model DFlash2.` \
  NAME="${NAME:-glm53-flash-c4-mtp}" \
  SERVED_MODEL_NAME=glm-5.3-flash \
  HOST="${HOST:-0.0.0.0}" \
  PORT="${PORT:-8080}" `# Port API.` \
  DOCKER_SUDO="${DOCKER_SUDO-}" \
  DRYRUN="${DRYRUN:-0}" `# 1 = tylko pokaz komende.` \
  REPLACE=0 `# Nie usuwaj istniejacego kontenera.` \
  BUILD_KERNELS=0 `# Uzyj juz zbudowanej biblioteki kerneli.` \
  MODEL_PREFLIGHT=r9700_vllm.models.glm5_next \
  GPUS=0,1,2,3,4,5,6,7 \
  TP=8 `# Model i wbudowana warstwa MTP pracuja na 8 GPU.` \
  MAXLEN="$MAXLEN" `# auto (-1): maks. JEDNEGO kontekstu; wejscie + wyjscie, takze thinking.` \
  NSEQ="$NSEQ" `# Domyslnie 4 aktywne zadania; auto nie gwarantuje 4 pelnych kontekstow naraz.` \
  NBT="${NBT:-2048}" `# Prefill i minimum cache encodera; 2048 pokrywa pelny limit obrazu.` \
  UTIL=0.90 `# Jawny KVMEM ma pierwszenstwo dla rozmiaru KV.` \
  KVMEM="${KVMEM:-4.125}" `# GiB KV na GPU, wspolne dla wszystkich zadan.` \
  KV_DTYPE=fp8_e4m3 `# Format KV modelu glownego.` \
  PREFIX_CACHE=1 `# APC: ponowne wykorzystanie wspolnych prefiksow.` \
  OFFLOAD_GB=0 `# Wagi i KV pozostaja na GPU; bez LMCache.` \
  MTP="$MTP" `# Wbudowane MTP z checkpointu glownego.` \
  EAGER=0 `# Uzywaj grafow, zamiast wymuszac eager.` \
  CGMODE=FULL_DECODE_ONLY `# Grafy tylko dla decode.` \
  CGSIZES="$graphs" `# MTP: rozmiary (MTP+1) razy liczba aktywnych sekwencji.` \
  REASONING_PARSER=glm45 `# Oddzielanie thinking od odpowiedzi.` \
  TOOL_CALL_PARSER=glm47 `# Parsowanie wywolan narzedzi.` \
  LMONLY= `# Multimodalny target; MTP draft nie obsluguje zewnetrznych embeddings. Vision niezakwalifikowane.` \
  `# Do 100 obrazow w calej historii zadania, bez wideo; do 2048 tokenow na obraz.` \
  `# max_pixels obejmuje 2 powtorzone klatki: 2 * 28 * 28 * 2048; zgodny budzet profilowania.` \
  EXTRA='--disable-custom-all-reduce --mm-encoder-attn-backend FLASH_ATTN --limit-mm-per-prompt {"image":100,"video":0} --mm-processor-kwargs {"max_image_tokens":2048,"max_pixels":3211264}' \
  ATTN= DRAFT_ATTN= CHAT_TEMPLATE= OVERLAYS= WRAP= \
  PROF=0 `# Profiler wylaczony.` \
  DOCKER_ARGS='-e FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE' `# Vision: Flash Attention Triton RDNA4; omija wadliwa sciezke Torch SDPA.` \
  P2P=1 `# Komunikacja GPU-GPU.` \
  HWQ=1 MWAITX=1 PIDNS=--pid=host \
  R9K_GLM_BASELINE=1 R9K_PLATFORM=1 VLLM_PLUGINS=r9700,r9700_glm \
  R9K_DISABLE=quant,models,mtp,gdn,attn `# Wylacz ogolne patche; aktywne sa adaptery GLM.` \
  VLLM_ROCM_USE_AITER=1 \
  R9K_GLM_FP8_SPARSE=1 `# Obsluga sparse attention z FP8 KV na RDNA4.` \
  R9K_GLM_DFLASH=0 `# Bez adaptera DFlash; uzyj wbudowanego GLM MTP vLLM.` \
  R9K_GLM_MOE=w4a4 `# Kernel ekspertow z wagami i aktywacjami MXFP4.` \
  R9K_GLM_INDEXER_GEMV=1 `# Zoptymalizowany GEMV indexera.` \
  R9K_GLM_INDEXER_WORKSPACE=bounded `# Bufor wg NSEQ i index_kpool=4.` \
  R9K_GLM_W4A4_MAX_ROWS=32 `# Limit wierszy dla szybkiej sciezki W4A4.` \
  R9K_GLM_W4A4_BATCH=packed `# Kernel dla wielu wierszy bez rozpakowania calych wag.` \
  R9K_AG="${R9K_AG:-0}" `# RCCL all-gather baseline; 1 = kandydat upstream wymagajacy testu TP8.` \
  R9K_AG_MAX_KB="${R9K_AG_MAX_KB:-16}" `# Kandydat: male all-gather; wieksze zostaja na RCCL.` \
  R9K_GLM_DENSE_FP8="${R9K_GLM_DENSE_FP8:-0}" `# Opt-in Quark dense FP8, 1-32 wiersze; wymaga nowej biblioteki.` \
  R9K_ARN_MAX_KB=16 R9K_AR4=0 R9K_AR_QUANT=0 \
  R9K_EXPERT_CACHE_SLOTS=0 `# Bez cache ekspertow offloadowanych z CPU.` \
  R9K_TARGET_LMHEAD=stock R9K_DRAFT_LMHEAD=stock \
  "$@" \
  DRAFT= R9K_GLM_DFLASH=0 \
  MAXLEN="$MAXLEN" \
  bash "$serve_dir/serve.sh"
