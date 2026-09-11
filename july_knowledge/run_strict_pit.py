"""Official strict-PIT build runner.

This runner changes no strategy or score rule. It applies two implementation-only
adapters before the strict knowledge build:
1) numpy/pandas scalars are serialized as native JSON scalars;
2) YYQYX theme/ladder rows are parsed by semantic headings + stock URLs rather
   than fragile site-specific CSS class names.
"""
import json
import numpy as np
import build_knowledge_layer_strict_pit as strict
from robust_yyqyx_parser import robust_parse_yyqyx

_original_dumps = json.dumps

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

strict.json.dumps = _safe_dumps
strict.b.parse_yyqyx = robust_parse_yyqyx

if __name__ == "__main__":
    strict.main()
