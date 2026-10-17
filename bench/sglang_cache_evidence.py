"""Explicit native cache evidence: a replay with zero hits is not a GPU hit."""


def native_cache_source(cached_tokens, detail, *, allow_miss=False):
    if type(cached_tokens) is not int or cached_tokens < 0:
        raise ValueError("Missing or invalid cached token counter")
    if cached_tokens == 0:
        if detail not in (None, {"device": 0, "host": 0}):
            raise ValueError("Zero-hit counter conflicts with cache source details")
        if not allow_miss:
            raise ValueError("Required cache hit was not observed")
        return {"cache_hit_observed": False, "cache_source": "none",
                "device_cached_tokens": 0, "host_cached_tokens": 0}
    if not isinstance(detail, dict):
        raise ValueError("Nonzero cache hit has no source evidence")
    device, host = detail.get("device"), detail.get("host")
    if (type(device) is not int or type(host) is not int or min(device, host) < 0
            or device+host != cached_tokens):
        raise ValueError("Cache source counts do not reconcile")
    return {"cache_hit_observed": True,
            "cache_source": "mixed" if device and host else "device" if device else "host",
            "device_cached_tokens": device, "host_cached_tokens": host}
