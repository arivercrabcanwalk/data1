"""Official strict-PIT build runner.

This runner changes no strategy or score rule. It applies implementation/provenance
adapters before the strict knowledge build:
1) numpy/pandas scalars are serialized as native JSON scalars;
2) YYQYX theme/ladder rows are parsed by semantic headings + stock URLs rather
   than fragile site-specific CSS class names;
3) the 23 YYQYX pages are read from the preserved GitHub Actions artifact created
   by run 34588894613 (artifact 10194836931, digest recorded in workflow/provenance)
   when available. This makes the historical knowledge build deterministic and
   avoids current-site soft blocking changing historical inputs.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import build_knowledge_layer_strict_pit as strict
from robust_yyqyx_parser import robust_parse_yyqyx

_original_dumps = json.dumps
_original_fetch = strict.b.fetch_with_retry
_BOOT = Path(__file__).resolve().parent / "bootstrap_v1" / "raw_web"
_BOOT_RUN = 34588894613
_BOOT_ARTIFACT = 10194836931
_BOOT_DIGEST = "sha256:e38a64f339d094b47db05b782cfc0817a702e8abfa8852c8414b86b33a008ad5"

def _default_scalar(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

def _safe_dumps(*args, **kwargs):
    kwargs.setdefault("default", _default_scalar)
    return _original_dumps(*args, **kwargs)

def _preserved_fetch(url, sid, day, tries=4):
    if sid == "yyqyx_limitup":
        p = _BOOT / f"{day}_yyqyx_limitup.html"
        if p.exists():
            b = p.read_bytes()
            return {
                "date": day,
                "source_id": sid,
                "url": url,
                "ok": True,
                "status": 200,
                "error": None,
                "path": str(p),
                "sha256": hashlib.sha256(b).hexdigest(),
                "fetched_at_utc": "2026-09-11T10:23:00Z~10:24:00Z",
                "html": b.decode("utf-8", errors="replace"),
                "provenance": f"github_actions_run={_BOOT_RUN};artifact={_BOOT_ARTIFACT};digest={_BOOT_DIGEST}"
            }
    return _original_fetch(url, sid, day, tries=tries)

strict.json.dumps = _safe_dumps
strict.b.parse_yyqyx = robust_parse_yyqyx
strict.b.fetch_with_retry = _preserved_fetch

if __name__ == "__main__":
    strict.main()
