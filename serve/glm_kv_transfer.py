"""Render the opt-in GLM LMCache connector configuration, without GPU imports."""
import json
import sys


def config(port: str) -> dict:
    try:
        number = int(port)
        if not 1 <= number <= 65535:
            raise ValueError
    except ValueError:
        raise ValueError("KV_CACHE_PORT must be an integer in 1..65535") from None
    return {
        "kv_connector": "GlmLMCacheMPConnector",
        # Local adapter handles GLM's circular scratch and kernel-paged MLA.
        "kv_connector_module_path": "r9700_vllm.compat.glm_lmcache",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "lmcache.mp.host": "tcp://127.0.0.1",
            "lmcache.mp.port": number,
            "lmcache.mp.mq_timeout": 30,
        },
    }


if __name__ == "__main__":
    try:
        print(json.dumps(config(sys.argv[1])))
    except (ValueError, IndexError) as exc:
        sys.exit(str(exc) or "KV_CACHE_PORT is required")
