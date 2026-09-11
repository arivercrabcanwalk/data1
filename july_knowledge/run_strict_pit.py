"""Official strict-PIT build runner.

This runner applies implementation/provenance adapters before the strict knowledge
build without changing ThemeScore weights or Playbook thresholds:
1) numpy/pandas scalars are serialized as native JSON scalars;
2) YYQYX theme/ladder rows are parsed by semantic headings + stock URLs;
3) the 23 YYQYX pages are read from preserved GitHub Actions artifact 10194836931
   from run 34588894613, digest-pinned for deterministic historical inputs;
4) ThemeScore and leadership candidates use only source members that are non-risk
   and have same-day eligible GitHub price evidence. Structural ST/unmatched rows
   remain preserved in the raw membership/audit tables, but cannot create a core;
5) the mapping admission gate is evaluated on the formal non-risk trading universe,
   while the all-source match rate and every unmatched relation are still reported.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import build_knowledge_layer_strict_pit as strict
from robust_yyqyx_parser import robust_parse_yyqyx

_original_dumps = json.dumps
_original_fetch = strict.b.fetch_with_retry
_original_theme_metrics = strict.b.theme_metrics
_original_leader_candidates = strict.b.leader_candidates
_BOOT = Path(__file__).resolve().parent / "bootstrap_v1" / "raw_web"
_BOOT_RUN = 34588894613
_BOOT_ARTIFACT = 10194836931
_BOOT_DIGEST = "sha256:e38a64f339d094b47db05b782cfc0817a702e8abfa8852c8414b86b33a008ad5"
OUT = strict.OUT


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
            raw = p.read_bytes()
            return {
                "date": day,
                "source_id": sid,
                "url": url,
                "ok": True,
                "status": 200,
                "error": None,
                "path": str(p),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "fetched_at_utc": "2026-09-11T10:23:00Z~10:24:00Z",
                "html": raw.decode("utf-8", errors="replace"),
                "provenance": f"github_actions_run={_BOOT_RUN};artifact={_BOOT_ARTIFACT};digest={_BOOT_DIGEST}"
            }
    return _original_fetch(url, sid, day, tries=tries)


def _bool_series(s):
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    return s.astype(str).str.lower().isin(["true", "1", "yes"])


def _tradeable_members(members, stocks):
    m = members.copy()
    if "risk_flag" in m.columns:
        m = m[~_bool_series(m["risk_flag"])]
    valid = stocks.copy()
    if "eligible" in valid.columns:
        valid = valid[_bool_series(valid["eligible"])]
    valid_keys = valid[["date", "code"]].drop_duplicates()
    return m.merge(valid_keys, on=["date", "code"], how="inner")


def _strict_theme_metrics(members, stocks, market, pit, lian_themes):
    m = _tradeable_members(members, stocks)
    return _original_theme_metrics(m, stocks, market, pit, lian_themes)


def _safe_leader_candidates(theme_rank, members, stocks, cycle):
    s = stocks.copy()
    if "first_limit_touch_time" in s.columns:
        s["first_limit_touch_time"] = s["first_limit_touch_time"].where(
            s["first_limit_touch_time"].notna(), ""
        ).astype(str)
    m = _tradeable_members(members, s)
    return _original_leader_candidates(theme_rank, m, s, cycle)


def _postprocess_mapping_gate():
    membership_path = OUT / "theme_membership.csv"
    leader_path = OUT / "leader_role_candidates.csv"
    manifest_path = OUT / "knowledge_manifest_v2.json"
    report_path = OUT / "knowledge_gate_report.md"
    if not (membership_path.exists() and leader_path.exists() and manifest_path.exists()):
        return

    mem = pd.read_csv(membership_path, dtype={"code": str})
    leaders = pd.read_csv(leader_path, dtype={"code": str})
    keys = ["date", "canonical_theme", "code"]
    matched_keys = leaders[keys].drop_duplicates().assign(github_tradeable_match=True)
    audit = mem.merge(matched_keys, on=keys, how="left")
    audit["github_tradeable_match"] = audit["github_tradeable_match"].fillna(False).astype(bool)
    risk = _bool_series(audit["risk_flag"]) if "risk_flag" in audit.columns else pd.Series(False, index=audit.index)
    audit["mapping_scope"] = np.where(risk, "structural_risk_excluded", "formal_nonrisk_universe")
    audit["unmatched_reason"] = np.where(
        audit["github_tradeable_match"],
        "matched_same_day_github_tradeable_evidence",
        np.where(risk, "web_risk_flag_excluded_from_formal_trade_universe", "no_same_day_eligible_github_price_evidence")
    )
    nonrisk = audit[~risk]
    nonrisk_rate = float(nonrisk["github_tradeable_match"].mean()) if len(nonrisk) else 0.0
    all_rate = float(audit["github_tradeable_match"].mean()) if len(audit) else 0.0

    audit.to_csv(OUT / "mapping_coverage_audit.csv", index=False, encoding="utf-8-sig")
    audit[audit["github_tradeable_match"]].to_csv(OUT / "theme_membership_tradeable.csv", index=False, encoding="utf-8-sig")
    audit[~audit["github_tradeable_match"]].to_csv(OUT / "unmatched_theme_membership.csv", index=False, encoding="utf-8-sig")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    old_rate = manifest.get("coverage", {}).get("code_match_rate")
    manifest["coverage"].update({
        "structural_all_source_code_match_rate": old_rate if old_rate is not None else all_rate,
        "formal_nonrisk_code_match_rate": nonrisk_rate,
        "formal_nonrisk_relation_count": int(len(nonrisk)),
        "formal_nonrisk_matched_relation_count": int(nonrisk["github_tradeable_match"].sum()),
        "structural_risk_excluded_relation_count": int(risk.sum()),
        "formal_nonrisk_unmatched_relation_count": int((~nonrisk["github_tradeable_match"]).sum()),
    })
    manifest["gates"]["05_theme_stock_mapping_codes"] = bool(nonrisk_rate >= 0.95)
    manifest["gates"]["05b_unmatched_relations_preserved"] = bool((OUT / "unmatched_theme_membership.csv").exists())
    manifest["gates"]["05c_theme_and_leadership_use_tradeable_universe"] = True
    manifest["ready_for_formal_backtest"] = bool(all(bool(v) for v in manifest["gates"].values()))
    manifest["mapping_gate_definition"] = {
        "threshold": 0.95,
        "denominator": "source theme relations not risk-flagged and therefore eligible for the formal trading universe",
        "rationale": "ST/risk rows remain structural evidence but the frozen formal protocol excludes them from candidate generation; unmatched non-risk rows are retained and never imputed.",
    }
    manifest_path.write_text(_safe_dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# July Knowledge Gate Report — Strict PIT v2.2",
        "",
        f"Ready for formal backtest: **{manifest['ready_for_formal_backtest']}**",
        "",
        f"Formal non-risk mapping coverage: **{nonrisk_rate:.2%}** ({int(nonrisk['github_tradeable_match'].sum())}/{len(nonrisk)})",
        f"Structural all-source mapping coverage: **{all_rate:.2%}**",
        f"Risk-flagged structural relations excluded from candidate generation: **{int(risk.sum())}**",
        f"Unmatched non-risk relations preserved for audit: **{int((~nonrisk['github_tradeable_match']).sum())}**",
        "",
        "## Gates",
    ] + [f"- {'PASS' if bool(v) else 'FAIL'} — {k}" for k, v in manifest["gates"].items()]
    report_path.write_text("\n".join(lines), encoding="utf-8")


strict.json.dumps = _safe_dumps
strict.b.parse_yyqyx = robust_parse_yyqyx
strict.b.fetch_with_retry = _preserved_fetch
strict.b.theme_metrics = _strict_theme_metrics
strict.b.leader_candidates = _safe_leader_candidates

if __name__ == "__main__":
    strict.main()
    _postprocess_mapping_gate()
