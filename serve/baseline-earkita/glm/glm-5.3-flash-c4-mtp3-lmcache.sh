#!/bin/bash
# Experimental GLM-5.3-Flash C4 + checkpoint MTP + LMCache, 8x R9700.
# MTP_DEPTH=3 remains the comparison baseline; depths 4/5 are opt-in.
# Nadpisania KEY=value jak w flashnext.sh. Podglad: DRYRUN=1.
# Jeden start: osobna usluga LMCache, potem GLM przez wspolny serve.sh.
# Nie uruchamiaj rownolegle z K4; kandydat wymaga testow poprawnosci.
# Nie zastepuje istniejacych kontenerow. DRYRUN=1 pokazuje obie komendy.

# Rozmiary grafow musza odpowiadac liczbie sekwencji; uwzglednij tez argumenty CLI.
NSEQ=${NSEQ:-4}
MAXLEN=${MAXLEN:-auto}
KV_OFFLOAD=${KV_OFFLOAD:-lmcache} # 0 = APC w VRAM; lmcache = eksperymentalny RAM + NVMe.
REFERENCE=${REFERENCE:-0} # 1 = target-only kontrola poprawnosci, bez spekulacji.
MTP_DEPTH=${MTP_DEPTH:-3} # 3 = sprawdzony kandydat; 4/5 = odrebne eksperymenty.
KV_CACHE_RAM_GB=${KV_CACHE_RAM_GB:-128}
KV_CACHE_DISK_GB=${KV_CACHE_DISK_GB:-512}
KV_CACHE_START_TIMEOUT=${KV_CACHE_START_TIMEOUT:-600} # Pelna alokacja duzej puli RAM moze trwac dluzej niz minute.
VLLM_SERVER_DEV_MODE=${VLLM_SERVER_DEV_MODE:-0} # 1 tylko dla izolowanego benchmarku z resetem APC.
for arg in "$@"; do
  case "$arg" in
    NSEQ=*) NSEQ=${arg#*=} ;;
    MAXLEN=*) MAXLEN=${arg#*=} ;;
    KV_OFFLOAD=*) KV_OFFLOAD=${arg#*=} ;;
    REFERENCE=*) REFERENCE=${arg#*=} ;;
    MTP_DEPTH=*) MTP_DEPTH=${arg#*=} ;;
    IMG=*|NAME=*|DRYRUN=*|DOCKER_SUDO=*|GPUS=*|KV_CACHE_*=*) export "$arg" ;;
    MTP=*|DRAFT=*|SPEC=*|SPEC_METHOD=*|DRAFT_KV_DTYPE=*|SPEC_EXTRA=*)
      echo 'Set MTP_DEPTH=3, 4 or 5; use REFERENCE=1 only for target-only control.' >&2
      exit 2 ;;
  esac
done
case "$MTP_DEPTH" in
  3|4|5) ;;
  *) echo 'MTP_DEPTH must be 3, 4 or 5.' >&2; exit 2 ;;
esac
if [ "$REFERENCE" = 1 ] && [ "$MTP_DEPTH" != 3 ]; then
  echo 'Target-only reference retains the MTP3 comparison geometry.' >&2
  exit 2
fi
case "$REFERENCE" in
  0) default_name=glm53-flash-c4-mtp3-lmcache
     default_cache_name=glm53-kv-cache-mtp3
     default_namespace=glm53-e975-fp8-mtp3-v1
     default_port=8081 default_cache_port=5556 default_metrics_port=9092 mtp_steps=3 ;;
  1) default_name=glm53-flash-c4-target-lmcache
     default_cache_name=glm53-kv-cache-target
     default_namespace=glm53-e975-fp8-target-v1
     default_port=8082 default_cache_port=5557 default_metrics_port=9093 mtp_steps= ;;
  *) echo 'REFERENCE must be 0 (MTP3) or 1 (target-only correctness control).' >&2; exit 2 ;;
esac
if [ "$REFERENCE" = 0 ] && [ "$MTP_DEPTH" != 3 ]; then
  default_name="glm53-flash-c4-mtp${MTP_DEPTH}-lmcache"
  default_cache_name="glm53-kv-cache-mtp${MTP_DEPTH}"
  default_namespace="glm53-e975-fp8-mtp${MTP_DEPTH}-v1"
  default_port="808${MTP_DEPTH}"
  default_cache_port="556${MTP_DEPTH}"
  default_metrics_port="910${MTP_DEPTH}"
  mtp_steps=$MTP_DEPTH
fi
KV_CACHE_NAME=${KV_CACHE_NAME:-$default_cache_name}
KV_CACHE_PORT=${KV_CACHE_PORT:-$default_cache_port}
KV_CACHE_NAMESPACE=${KV_CACHE_NAMESPACE:-$default_namespace}
KV_CACHE_METRICS_PORT=${KV_CACHE_METRICS_PORT:-$default_metrics_port}
[[ "$KV_CACHE_START_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || { echo 'KV_CACHE_START_TIMEOUT must be positive seconds.' >&2; exit 2; }
case "$VLLM_SERVER_DEV_MODE" in
  0) dev_docker_arg= ;;
  1)
    [ "${HOST:-0.0.0.0}" = 127.0.0.1 ] || { echo 'VLLM_SERVER_DEV_MODE=1 requires HOST=127.0.0.1.' >&2; exit 2; }
    dev_docker_arg=' -e VLLM_SERVER_DEV_MODE=1' ;;
  *) echo 'VLLM_SERVER_DEV_MODE must be 0 or 1.' >&2; exit 2 ;;
esac
case "$NSEQ" in
  1|2|4)
    graph_rows=$((mtp_steps + 1))
    [ "$REFERENCE" = 0 ] || graph_rows=4
    graphs=$graph_rows
    for ((row=2; row<=NSEQ; row++)); do
      graphs+=",$((row * graph_rows))"
    done ;;
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
  IMG="${IMG:-r9700/vllm:glm53-lmcache055-mtp5004}" `# Ten sam przypiety vLLM/LMCache; tylko backport relokacji MTP.` \
  MODELS_DIR="${MODELS_DIR:-/mnt/ai/models/glm}" `# Host -> /models w kontenerze.` \
  MODEL="${MODEL:-/models/GLM-5.3-Flash-Quark-MXFP4}" `# Model glowny, eksperci MXFP4.` \
  DRAFT= `# Jedna oryginalna warstwa NextN z checkpointu; bez DFlash2.` \
  NAME="${NAME:-$default_name}" \
  SERVED_MODEL_NAME=glm-5.3-flash \
  HOST="${HOST:-0.0.0.0}" \
  PORT="${PORT:-$default_port}" `# Oddzielny port; nie przejmuj portu K4.` \
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
  PREFIX_CACHE=1 `# APC: ponowne wykorzystanie wspolnych prefiksow.` \
  KV_CACHE_DIR="${KV_CACHE_DIR-/mnt/ai/cache/}" `# Katalog NVMe; ten sam podaj osobnemu launcherowi glm-kv-cache.sh.` \
  OFFLOAD_GB=0 `# Wagi pozostaja na GPU; KV_OFFLOAD dotyczy tylko cache prefiksow.` \
  MTP="$mtp_steps" `# Checkpoint NextN; puste w target-only kontroli.` \
  EAGER=0 `# Uzywaj grafow, zamiast wymuszac eager.` \
  CGMODE=FULL_DECODE_ONLY `# Grafy tylko dla decode.` \
  CGSIZES="$graphs" `# Capture sizes = (depth + 1) * concurrency.` \
  REASONING_PARSER=glm45 `# Oddzielanie thinking od odpowiedzi.` \
  TOOL_CALL_PARSER=glm47 `# Parsowanie wywolan narzedzi.` \
  LMONLY= `# Tekst + obrazy; fallback glm-5.3-flash.sh pozostaje tekstowy.` \
  `# Do 100 obrazow w calej historii zadania, bez wideo; do 2048 tokenow na obraz.` \
  `# max_pixels obejmuje 2 powtorzone klatki: 2 * 28 * 28 * 2048; zgodny budzet profilowania.` \
  EXTRA='--disable-custom-all-reduce --mm-encoder-attn-backend FLASH_ATTN --limit-mm-per-prompt {"image":100,"video":0} --mm-processor-kwargs {"max_image_tokens":2048,"max_pixels":3211264}' \
  ATTN= CHAT_TEMPLATE= OVERLAYS= WRAP= \
  PROF=0 `# Profiler wylaczony.` \
  DOCKER_ARGS="-e FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE${dev_docker_arg}" `# Vision: Flash Attention Triton RDNA4; dev mode tylko na loopback.` \
  P2P=1 `# Komunikacja GPU-GPU.` \
  HWQ=1 MWAITX=1 PIDNS=--pid=host \
  R9K_GLM_BASELINE=1 R9K_PLATFORM=1 VLLM_PLUGINS=r9700,r9700_glm \
  R9K_DISABLE=quant,models,mtp,gdn,attn `# Wylacz ogolne patche; aktywne sa adaptery GLM.` \
  VLLM_ROCM_USE_AITER=1 \
  R9K_GLM_FP8_SPARSE=1 `# Obsluga sparse attention z FP8 KV na RDNA4.` \
  R9K_GLM_DFLASH=0 `# Wbudowany MTP, bez adaptera DFlash.` \
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
  "$@" "${kv_args[@]}" DRAFT= MTP="$mtp_steps" R9K_GLM_DFLASH=0 CGSIZES="$graphs" \
  MAXLEN="$MAXLEN" \
  bash "$serve_dir/serve.sh"
}

# Cache i model musza miec ten sam obraz, mapowanie GPU i port transferu.
# Wariant KV_OFFLOAD=0 uruchamia sam model.
if [ "${DRYRUN:-0}" != 1 ] && curl -fsS --max-time 2 \
    http://127.0.0.1:8080/health >/dev/null 2>&1; then
  echo 'Another model is serving on port 8080 (possibly K4); stop the exclusive GPU run before starting MTP3. No services changed.' >&2
  exit 2
fi
if [ "$KV_OFFLOAD" = lmcache ]; then
  cache_args=("IMG=${IMG:-r9700/vllm:glm53-lmcache055-mtp5004}"
    "NAME=$KV_CACHE_NAME" "KV_CACHE_PORT=$KV_CACHE_PORT"
    "KV_CACHE_DIR=${KV_CACHE_DIR-/mnt/ai/cache/}"
    "KV_CACHE_NAMESPACE=$KV_CACHE_NAMESPACE"
    "KV_CACHE_METRICS_PORT=$KV_CACHE_METRICS_PORT"
    "KV_CACHE_RAM_GB=$KV_CACHE_RAM_GB" "KV_CACHE_DISK_GB=$KV_CACHE_DISK_GB"
    "GPUS=${GPUS:-0,1,2,3,4,5,6,7}" "DOCKER_SUDO=${DOCKER_SUDO-}")
  if [ "${DRYRUN:-0}" = 1 ]; then
    bash "$serve_dir/glm-kv-cache.sh" "${cache_args[@]}" DRYRUN=1 || exit $?
  else
    read -r -a docker_prefix <<< "${DOCKER_SUDO-}"
    docker_cmd=("${docker_prefix[@]}" docker)
    if "${docker_cmd[@]}" container inspect "${NAME:-$default_name}" >/dev/null 2>&1; then
      echo "Model container already exists: ${NAME:-$default_name}. No services changed." >&2
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
