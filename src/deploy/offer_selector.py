"""
Offer selection for Vast.ai — hard filters, scoring and ranking.

Replaces the old "take the single cheapest offer" behaviour in deploy.py with
a two-stage pipeline:

  1. build_search_query() — the hard numeric gates go into the
     `vastai search offers` query string, so the API only returns plausible
     machines in the first place.
  2. rank_offers() — Python-side re-check of EVERY gate (a query field the
     CLI silently ignores must not slip through), an outlier-price guard, then
     a lexicographic score:

         reliability (bucketed at 0.95) → min(inet_down, inet_up)
            → cpu_cores_effective → -price

     The deploy script rents the top-K (retry_count) in order, so a single
     flaky offer no longer kills the whole deployment.

All functions are pure (no torch / vastai / streamlit imports) so they are
unit-testable on any machine — see tests/test_offer_selector.py.
"""

from __future__ import annotations

import statistics
from typing import Any

# Defaults — mirrored in configs/default_config.yaml (mlops.vast.selector) so
# the Streamlit Cloud tab shows the same numbers. Every quality gate defaults
# to 0 / "" = NO filtering: the search returns whatever the market has, and the
# score (reliability → internet → CPU → price) picks the best of it. Raise a
# gate (UI / env / config) to exclude machines below that bar.
DEFAULTS: dict[str, Any] = {
    "pool_size": 50,            # top-N cheapest candidates fetched from the API
    "retry_count": 3,           # how many of the ranked pool to try renting
    # Quality gates default to 0 / "" = NO filtering. Every numeric gate has a
    # min and a max twin, so a range can be expressed (e.g. 300 ≤ download ≤ 2000).
    "min_reliability": 0.0,     # historical host uptime health, 0..1 (min only)
    "min_inet_down_mbps": 0,    # MB/s (vast reports internet speeds in MB/s)
    "max_inet_down_mbps": 0,
    "min_inet_up_mbps": 0,
    "max_inet_up_mbps": 0,
    "min_cpu_cores": 0,         # advertised vCPUs
    "max_cpu_cores": 0,
    "min_cpu_ram_gb": 0,        # host RAM (CLI query unit is GB)
    "max_cpu_ram_gb": 0,
    "min_disk_gb": 0,           # free disk space
    "max_disk_gb": 0,
    "min_duration_days": 0,     # host max rental length (hours in the query)
    "max_duration_days": 0,
    "min_price_per_hour": 0.0,  # $/h band (0 = open)
    "max_price_per_hour": 0.0,
    "min_pcie_bw_gbps": 0,      # CPU↔GPU PCIe bandwidth (GB/s)
    "max_pcie_bw_gbps": 0,
    "min_gpu_frac": 0.0,        # min GPU fraction (1.0 = full GPU only)
    "blocked_countries": "",    # comma-separated ISO codes, e.g. "US,HK"
    "pick_best": True,          # True = best by quality score; False = cheapest
}

# Gate applied ONLY in Python (never in the query): display-attached GPUs often
# run locked clocks and are not worth renting. Fractional GPUs are handled by
# the configurable min_gpu_frac gate above (default off).
_REJECT_DISPLAY_GPU = True

# Scoring knobs (see score_offer / rank_offers):
_RELIABILITY_GOOD = 0.95      # hosts at/above this are "reliable" → bucket to 1.0
_PRICE_GUARD_MULTIPLE = 3.0   # drop offers pricier than 3× the pool median price
_MIN_POOL_FOR_PRICE_GUARD = 5 # only apply the price guard on pools of ≥ this size


def _num(offer: dict, key: str, default: float = 0.0) -> float:
    """Float-cast an offer field, falling back to `default` on missing/garbage."""
    try:
        return float(offer.get(key, default))
    except (TypeError, ValueError):
        return default


def _blocked_set(sel: dict) -> set[str]:
    raw = str(sel.get("blocked_countries") or "")
    return {c.strip().upper() for c in raw.split(",") if c.strip()}


def _fmt_qt(value: float) -> str:
    """
    Format a query value for the vastai CLI, which type-checks strictly:
    integer fields (cpu_cores, cpu_ram, disk_space, duration, inet_* …) reject
    float literals like `8.0`. Integral values render as ints; only genuine
    fractions (e.g. reliability 0.95) keep decimals.
    """
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def build_search_query(gpu_target: str, min_cuda_version: str,
                       sel: dict | None = None) -> str:
    """
    Build the `vastai search offers` query string for the hard gates.

    The CUDA gate (cuda_vers>=...) is passed through untouched — the caller
    owns it (uv.lock pins a CUDA 13.x torch build, so this must stay >= 13).
    Fields whose value is 0 are omitted (no filter).
    """
    s = {**DEFAULTS, **(sel or {})}
    parts = [f"gpu_name={gpu_target}", "num_gpus=1",
             f"cuda_vers>={min_cuda_version}"]

    # (field, op, value) — min gates use >=, max gates use <=; 0 = no filter.
    query_gates = [
        ("reliability", ">=", float(s["min_reliability"])),
        ("inet_down", ">=", float(s["min_inet_down_mbps"])),
        ("inet_down", "<=", float(s["max_inet_down_mbps"])),
        ("inet_up", ">=", float(s["min_inet_up_mbps"])),
        ("inet_up", "<=", float(s["max_inet_up_mbps"])),
        ("cpu_cores", ">=", float(s["min_cpu_cores"])),
        ("cpu_cores", "<=", float(s["max_cpu_cores"])),
        # Unit quirks of the `vastai search offers` CLI (differs from the REST
        # response, where cpu_ram is MB and duration is seconds):
        #   cpu_ram  → GB
        #   duration → hours
        ("cpu_ram", ">=", float(s["min_cpu_ram_gb"])),
        ("cpu_ram", "<=", float(s["max_cpu_ram_gb"])),
        ("disk_space", ">=", float(s["min_disk_gb"])),
        ("disk_space", "<=", float(s["max_disk_gb"])),
        ("duration", ">=", float(s["min_duration_days"]) * 24),
        ("duration", "<=", float(s["max_duration_days"]) * 24),
        ("dph_total", ">=", float(s["min_price_per_hour"])),
        ("dph_total", "<=", float(s["max_price_per_hour"])),
    ]
    for field, op, val in query_gates:
        if val > 0:
            parts.append(f"{field}{op}{_fmt_qt(val)}")
    return " ".join(parts)


def _geo_matches(offer: dict, blocked: set[str]) -> bool:
    """
    True if the offer's geolocation matches a blocked entry.

    vast reports geolocation as "Region, CC" (e.g. "California, US") — so
    blocking "US" must match the trailing country code, not just the whole
    string. Exact full-string matches (e.g. "California, US") also work.
    """
    geo = str(offer.get("geolocation") or "").strip().upper()
    if not geo:
        return False
    if geo in blocked:
        return True
    return geo.rsplit(",", 1)[-1].strip() in blocked


def passes_filters(offer: dict, sel: dict, blocked: set[str]) -> bool:
    """True if the offer satisfies every configured gate (pure, total)."""
    s = {**DEFAULTS, **(sel or {})}

    if float(s["min_reliability"]) > 0 \
            and _num(offer, "reliability") < float(s["min_reliability"]):
        return False
    if float(s["min_inet_down_mbps"]) > 0 \
            and _num(offer, "inet_down") < float(s["min_inet_down_mbps"]):
        return False
    if float(s["max_inet_down_mbps"]) > 0 \
            and _num(offer, "inet_down") > float(s["max_inet_down_mbps"]):
        return False
    if float(s["min_inet_up_mbps"]) > 0 \
            and _num(offer, "inet_up") < float(s["min_inet_up_mbps"]):
        return False
    if float(s["max_inet_up_mbps"]) > 0 \
            and _num(offer, "inet_up") > float(s["max_inet_up_mbps"]):
        return False
    if float(s["min_cpu_cores"]) > 0:
        if _num(offer, "cpu_cores") < float(s["min_cpu_cores"]):
            return False
        # effective cores reflect host contention; when the field is present it
        # must also clear the bar (absent -> rely on the advertised count).
        eff = _num(offer, "cpu_cores_effective")
        if eff > 0 and eff < float(s["min_cpu_cores"]):
            return False
    if float(s["max_cpu_cores"]) > 0 \
            and _num(offer, "cpu_cores") > float(s["max_cpu_cores"]):
        return False
    if float(s["min_cpu_ram_gb"]) > 0 \
            and _num(offer, "cpu_ram") < float(s["min_cpu_ram_gb"]) * 1024:
        return False
    if float(s["max_cpu_ram_gb"]) > 0 \
            and _num(offer, "cpu_ram") > float(s["max_cpu_ram_gb"]) * 1024:
        return False
    if float(s["min_disk_gb"]) > 0 \
            and _num(offer, "disk_space") < float(s["min_disk_gb"]):
        return False
    if float(s["max_disk_gb"]) > 0 \
            and _num(offer, "disk_space") > float(s["max_disk_gb"]):
        return False
    if float(s["min_duration_days"]) > 0 \
            and _num(offer, "duration") < float(s["min_duration_days"]) * 86400:
        return False
    if float(s["max_duration_days"]) > 0 \
            and _num(offer, "duration") > float(s["max_duration_days"]) * 86400:
        return False
    if float(s["min_price_per_hour"]) > 0 \
            and _num(offer, "dph_total") < float(s["min_price_per_hour"]):
        return False
    if float(s["max_price_per_hour"]) > 0 \
            and _num(offer, "dph_total") > float(s["max_price_per_hour"]):
        return False
    if float(s["min_pcie_bw_gbps"]) > 0 \
            and _num(offer, "pcie_bw") < float(s["min_pcie_bw_gbps"]):
        return False
    if float(s["max_pcie_bw_gbps"]) > 0 \
            and _num(offer, "pcie_bw") > float(s["max_pcie_bw_gbps"]):
        return False
    if float(s["min_gpu_frac"]) > 0 \
            and _num(offer, "gpu_frac") < float(s["min_gpu_frac"]):
        return False
    if _REJECT_DISPLAY_GPU and _num(offer, "gpu_display_active") > 0:
        return False
    if blocked and _geo_matches(offer, blocked):
        return False
    return True


def score_offer(offer: dict) -> tuple:
    """
    Lexicographic score — higher is better.

    (reliability-bucketed, min(inet_down, inet_up), cpu_cores_effective, -price).

    Reliability is bucketed at 0.95 so the tiny float differences near 1.0
    (0.9999 vs 0.9997) do NOT swamp real differences in internet/CPU — every
    host that is genuinely reliable ranks equally, then the fastest internet
    wins, then the strongest CPU, and price only breaks ties. Hosts below 0.95
    rank below all reliable ones (raw score), which keeps the "server dies
    mid-run" problem rare.
    """
    rel = _num(offer, "reliability")
    rel_bucket = 1.0 if rel >= _RELIABILITY_GOOD else rel
    inet = min(_num(offer, "inet_down"), _num(offer, "inet_up"))
    cpu = _num(offer, "cpu_cores_effective") or _num(offer, "cpu_cores")
    price = _num(offer, "dph_total") or _num(offer, "dph")
    return (rel_bucket, inet, cpu, -price)


def rank_offers(offers: list[dict], sel: dict | None = None,
                mode: str = "best") -> list[dict]:
    """
    Filter by every gate, drop outlier-priced offers, then sort.

    The price guardrail is data-driven: on pools of ≥ 5 offers, anything priced
    above 3× the pool median is dropped before sorting, so a single overpriced
    machine can never win by accident.

    mode="best" (default): sort by quality score (score_offer) descending —
    the best candidate first. mode="cheapest": sort by price ascending — the
    cheapest offer that passes the gates first (used when pick_best is off).
    """
    s = {**DEFAULTS, **(sel or {})}
    blocked = _blocked_set(s)
    ok = [o for o in offers if passes_filters(o, s, blocked)]
    prices = [_num(o, "dph_total") for o in ok if _num(o, "dph_total") > 0]
    if len(ok) >= _MIN_POOL_FOR_PRICE_GUARD and prices:
        cap = _PRICE_GUARD_MULTIPLE * statistics.median(prices)
        ok = [o for o in ok if _num(o, "dph_total") <= cap]
    if mode == "cheapest":
        ok.sort(key=lambda o: _num(o, "dph_total"))
    else:
        ok.sort(key=score_offer, reverse=True)
    return ok


def describe_offer(offer: dict) -> str:
    """One-line human summary of an offer (for deploy logs)."""
    cpu = _num(offer, "cpu_cores_effective") or _num(offer, "cpu_cores")
    return (
        f"#{offer.get('id')} {offer.get('gpu_name')} "
        f"${_num(offer, 'dph_total'):.3f}/h "
        f"rel={_num(offer, 'reliability'):.2f} "
        f"net={_num(offer, 'inet_down'):.0f}/{_num(offer, 'inet_up'):.0f} MB/s "
        f"cpu={cpu:.0f} geo={offer.get('geolocation')}"
    )
