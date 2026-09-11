"""Official strict-PIT build runner.

This wrapper changes serialization only: numpy/pandas scalar values are converted
to native JSON scalars when the knowledge manifest is written. It does not alter
any market, theme, leadership, playbook, or gate rule.
"""
import json
import numpy as np
import build_knowledge_layer_strict_pit as strict

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

if __name__ == "__main__":
    strict.main()
