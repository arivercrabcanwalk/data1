"""Strict PIT entrypoint.

The full v2 builder knows how to parse Lianban historical pages, but their current
archive form does not prove what exact content was available on each historical
close. For the formal knowledge gate we therefore do not fetch/use Lianban live.
Timestamped same-day corroboration is the semantic PIT source; YYQYX is retained
as a structural limit-up/theme-membership archive and all price facts are checked
against GitHub minute bars.
"""
import build_knowledge_layer_v2 as b

_original = b.fetch_with_retry

def strict_fetch(url, sid, day, tries=4):
    if sid == "lianban_archive":
        return {
            "date": day,
            "source_id": sid,
            "url": url,
            "ok": False,
            "status": None,
            "error": "SKIPPED_STRICT_PIT: archive publication/version timing not proven; retained only as optional research reference"
        }
    return _original(url, sid, day, tries=tries)

b.fetch_with_retry = strict_fetch

if __name__ == "__main__":
    b.main()
