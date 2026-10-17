#!/bin/bash
# GLM-5.3-Flash C4 + DFlash K4 + LMCache RAM/NVMe, 8x R9700.
# Nadpisania KEY=value jak w flashnext.sh. Podglad: DRYRUN=1.
# Jeden start: usluga LMCache, potem GLM przez wspolny serve.sh.
# Wyniki C1/C2/C4 32K/128K low i parametry 128/512 GiB: glm-5.3-flash-c4-lmcache.md.
# Nie zastepuje istniejacych kontenerow. DRYRUN=1 pokazuje obie komendy.

# Rozmiary grafow musza odpowiadac liczbie sekwencji; uwzglednij tez argumenty CLI.
NSEQ=${NSEQ:-4}
MAXLEN=${MAXLEN:-auto}
KV_OFFLOAD=${KV_OFFLOAD:-lmcache} # 0 = APC w VRAM; lmcache = eksperymentalny RAM + NVMe.
KV_CACHE_NAME=${KV_CACHE_NAME:-glm53-kv-cache-c4} # Nazwa osobnego kontenera cache.
KV_CACHE_PORT=${KV_CACHE_PORT:-5555} # Port osobno uruchamianego serve/glm-kv-cache.sh.
KV_CACHE_REUSE=${KV_CACHE_REUSE:-0} # 1 = uzyj juz dzialajacego cache; nie alokuj drugiej puli RAM.
KV_CACHE_START_TIMEOUT=${KV_CACHE_START_TIMEOUT:-600} # Pelna alokacja duzej puli RAM moze trwac dluzej niz minute.
for arg in "$@"; do
  case "$arg" in
    NSEQ=*) NSEQ=${arg#*=} ;;
    MAXLEN=*) MAXLEN=${arg#*=} ;;
    KV_OFFLOAD=*) KV_OFFLOAD=${arg#*=} ;;
    IMG=*|NAME=*|DRYRUN=*|DOCKER_SUDO=*|GPUS=*|KV_CACHE_*=*) export "$arg" ;;
  esac
done
[[ "$KV_CACHE_START_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || { echo 'KV_CACHE_START_TIMEOUT must be positive seconds.' >&2; exit 2; }
[[ "$KV_CACHE_REUSE" = 0 || "$KV_CACHE_REUSE" = 1 ]] || { echo 'KV_CACHE_REUSE must be 0 or 1.' >&2; exit 2; }
case "$NSEQ" in
  1) graphs=5 ;;
  2) graphs=5,10 ;;
  4) graphs=5,10,15,20 ;;
  *) echo 'NSEQ must be 1, 2 or 4.' >&2; exit 2 ;;
esac
[ "$MAXLEN" = auto ] && MAXLEN=-1
serve_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
kv_args=()
case "$KV_OFFLOAD" in
  0) ;;
  lmcache)
    kv_config=$(python3 "$serve_dir/glm_kv_transfer.py" "$KV_CACHE_PORT") || exit $?
    kv_args=("KV_TRANSFER_CONFIG=$kv_config")
    ;;
  *) echo 'KV_OFFLOAD must be 0 or lmcache.' >&2; exit 2 ;;
esac

launch_model() {
env \
  REPO="${REPO:-$(dirname "$serve_dir")}" `# Katalog pluginu na hoscie.` \
  IMG="${IMG:-r9700/vllm:glm53-lmcache055}" `# Obraz z przypieta wersja vLLM.` \
  MODELS_DIR="${MODELS_DIR:-/mnt/ai/models/glm}" `# Host -> /models w kontenerze.` \
  MODEL="${MODEL:-/models/GLM-5.3-Flash-Quark-MXFP4}" `# Model glowny, eksperci MXFP4.` \
  DRAFT="${DRAFT:-/models/GLM-5.3-Flash-DFlash2-HF-bf582e4}" `# Drafter, wagi BF16.` \
  NAME="${NAME:-glm53-flash-c4-lmcache}" \
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
  NSEQ="$NSEQ" `# Domyslnie 4 aktywne zadania; auto nie gwarantuje 4 pelnych kontekstow naraz.` \
  NBT="${NBT:-2048}" `# Prefill i minimum cache encodera; 2048 pokrywa pelny limit obrazu.` \
  UTIL=0.90 `# Jawny KVMEM ma pierwszenstwo dla rozmiaru KV.` \
  KVMEM="${KVMEM:-4.125}" `# GiB KV na GPU, wspolne dla wszystkich zadan.` \
  KV_DTYPE=fp8_e4m3 `# Format KV modelu glownego.` \
  DRAFT_KV_DTYPE=fp8_e4m3 `# Format KV draftera; jego wagi pozostaja BF16.` \
  PREFIX_CACHE=1 `# APC: ponowne wykorzystanie wspolnych prefiksow.` \
  KV_CACHE_DIR="${KV_CACHE_DIR-/mnt/ai/cache/}" `# Katalog NVMe; ten sam podaj osobnemu launcherowi glm-kv-cache.sh.` \
  OFFLOAD_GB=0 `# Wagi pozostaja na GPU; KV_OFFLOAD dotyczy tylko cache prefiksow.` \
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
  R9K_AG="${R9K_AG:-0}" `# RCCL all-gather baseline; 1 = kandydat upstream wymagajacy testu TP8.` \
  R9K_AG_MAX_KB="${R9K_AG_MAX_KB:-16}" `# Kandydat: male all-gather; wieksze zostaja na RCCL.` \
  R9K_GLM_DENSE_FP8="${R9K_GLM_DENSE_FP8:-0}" `# Opt-in Quark dense FP8, 1-32 wiersze; wymaga nowej biblioteki.` \
  R9K_ARN_MAX_KB=16 R9K_AR4=0 R9K_AR_QUANT=0 \
  R9K_EXPERT_CACHE_SLOTS=0 `# Bez cache ekspertow offloadowanych z CPU.` \
  R9K_TARGET_LMHEAD=stock R9K_DRAFT_LMHEAD=stock \
  "$@" "${kv_args[@]}" \
  MAXLEN="$MAXLEN" \
  bash "$serve_dir/serve.sh"
}

# Cache i model musza miec ten sam obraz, mapowanie GPU i port transferu.
# Wariant KV_OFFLOAD=0 uruchamia sam model.
if [ "$KV_OFFLOAD" = lmcache ]; then
  cache_args=("IMG=${IMG:-r9700/vllm:glm53-lmcache055}"
    "NAME=$KV_CACHE_NAME" "KV_CACHE_PORT=$KV_CACHE_PORT"
    "KV_CACHE_DIR=${KV_CACHE_DIR-/mnt/ai/cache/}"
    "GPUS=${GPUS:-0,1,2,3,4,5,6,7}" "DOCKER_SUDO=${DOCKER_SUDO-}")
  if [ "$KV_CACHE_REUSE" = 1 ]; then
    if [ "${DRYRUN:-0}" != 1 ]; then
      python3 - "$KV_CACHE_NAME" "$KV_CACHE_PORT" "${IMG:-r9700/vllm:glm53-lmcache055}" <<'PY' || exit $?
import json, subprocess, sys, urllib.request
name, port, image = sys.argv[1:]
cache = json.loads(subprocess.check_output(['docker', 'inspect', name]))[0]
image_id = json.loads(subprocess.check_output(['docker', 'image', 'inspect', image]))[0]['Id']
args = cache['Config']['Cmd']
def option(key):
    return args[args.index(key) + 1] if key in args else None
if (not cache['State']['Running'] or cache['Image'] != image_id
        or cache['HostConfig']['NetworkMode'] != 'host'
        or option('--port') != port or option('--chunk-size') != '1152'):
    raise SystemExit('Existing LMCache container/image/port/chunk geometry does not match this GLM profile')
with urllib.request.urlopen('http://127.0.0.1:9091/metrics', timeout=5) as response:
    if response.status != 200:
        raise SystemExit('Existing LMCache metrics endpoint is not ready')
print(f'Reusing LMCache {name}; no second host RAM pool allocated.')
PY
    else
      echo "DRYRUN: reuse running LMCache $KV_CACHE_NAME on port $KV_CACHE_PORT"
    fi
  elif [ "${DRYRUN:-0}" = 1 ]; then
    bash "$serve_dir/glm-kv-cache.sh" "${cache_args[@]}" DRYRUN=1 || exit $?
  else
    read -r -a docker_prefix <<< "${DOCKER_SUDO-}"
    docker_cmd=("${docker_prefix[@]}" docker)
    if "${docker_cmd[@]}" container inspect "${NAME:-glm53-flash-c4-lmcache}" >/dev/null 2>&1; then
      echo "Model container already exists: ${NAME:-glm53-flash-c4-lmcache}. No services changed." >&2
      exit 1
    fi
    cache_id=$(bash "$serve_dir/glm-kv-cache.sh" "${cache_args[@]}") || exit $?
    # Stop only the cache created by this invocation if launching fails.
    cleanup_failed_start() {
      status=$?
      if [ "$status" -ne 0 ]; then
        "${docker_cmd[@]}" stop --time 5 "$cache_id" >&2 || true
      fi
    }
    trap cleanup_failed_start EXIT
    ready=0
    for ((attempt=0; attempt<KV_CACHE_START_TIMEOUT; attempt++)); do
      state=$("${docker_cmd[@]}" inspect --format '{{.State.Running}}' "$cache_id") || break
      [ "$state" = true ] || break
      if "${docker_cmd[@]}" logs "$cache_id" 2>&1 | grep -Fq "LMCache ZMQ cache server is running on tcp://127.0.0.1:$KV_CACHE_PORT"; then
        ready=1
        break
      fi
      sleep 1
    done
    if [ "$ready" != 1 ]; then
      "${docker_cmd[@]}" logs --tail 30 "$cache_id" >&2
      echo 'LMCache failed to become ready; GLM was not started.' >&2
      exit 1
    fi
    echo "LMCache ready: $KV_CACHE_NAME; starting GLM (model warmup follows)."
  fi
fi
launch_model "$@"
