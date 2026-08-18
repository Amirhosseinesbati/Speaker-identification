"""
Tests for `src/deploy/offer_selector.py` — the Vast.ai offer filters, scoring
and ranking used by deploy.py (pure functions, no torch / vastai / streamlit).

DEFAULTS intentionally apply NO filters (every gate is 0 / "") so the search
takes whatever the market has; the strict gates in these tests are therefore
passed EXPLICITLY in the selector dicts.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from src.deploy import offer_selector as os_

# Offers crafted to trip exactly one gate each under the strict selector below.
OFFERS = [
    {"id": 1, "dph_total": 0.25, "reliability": 0.99,
     "inet_down": 500, "inet_up": 200, "cpu_cores": 16,
     "cpu_cores_effective": 16, "cpu_ram": 65536, "disk_space": 200,
     "duration": 7 * 86400, "gpu_frac": 1.0, "gpu_display_active": False,
     "pcie_bw": 32, "geolocation": "NL"},                       # all-good
    {"id": 2, "dph_total": 0.20, "reliability": 0.80,          # low reliability
     "inet_down": 900, "inet_up": 300, "cpu_cores": 32,
     "cpu_cores_effective": 32, "cpu_ram": 131072, "disk_space": 500,
     "duration": 30 * 86400, "gpu_frac": 1.0, "gpu_display_active": False,
     "pcie_bw": 32, "geolocation": "US"},
    {"id": 3, "dph_total": 0.30, "reliability": 0.97,
     "inet_down": 100, "inet_up": 50, "cpu_cores": 8,          # slow internet
     "cpu_cores_effective": 8, "cpu_ram": 32768, "disk_space": 150,
     "duration": 7 * 86400, "gpu_frac": 1.0, "gpu_display_active": False,
     "pcie_bw": 16, "geolocation": "HK"},
    {"id": 4, "dph_total": 0.22, "reliability": 0.98,
     "inet_down": 600, "inet_up": 150, "cpu_cores": 12,
     "cpu_cores_effective": 12, "cpu_ram": 49152, "disk_space": 180,
     "duration": 5 * 86400, "gpu_frac": 0.5, "gpu_display_active": False,  # fractional
     "pcie_bw": 16, "geolocation": "DE"},
]

# The old recommended strict gates, now explicit (defaults are all off).
STRICT = {
    "min_reliability": 0.95, "min_inet_down_mbps": 300, "min_inet_up_mbps": 100,
    "min_cpu_cores": 8, "min_cpu_ram_gb": 32, "min_disk_gb": 128,
    "min_duration_days": 3, "min_pcie_bw_gbps": 16, "min_gpu_frac": 1.0,
}


def test_defaults_apply_no_filters():
    assert os_.DEFAULTS["min_reliability"] == 0
    assert os_.DEFAULTS["min_inet_down_mbps"] == 0
    assert os_.DEFAULTS["min_cpu_cores"] == 0
    assert os_.DEFAULTS["min_gpu_frac"] == 0
    assert os_.DEFAULTS["pick_best"] is True
    q = os_.build_search_query("RTX_3090", "13", os_.DEFAULTS)
    assert q == "gpu_name=RTX_3090 num_gpus=1 cuda_vers>=13"


def test_build_search_query_includes_gates_and_cuda_passthrough():
    q = os_.build_search_query("RTX_3090", "13", STRICT)
    # CLI unit quirks: cpu_ram is GB, duration is hours (the REST response uses
    # MB / seconds — Python-side gates keep those). Integral fields must render
    # as ints — the CLI type-checks strictly and rejects `8.0` for int keys.
    for piece in ["gpu_name=RTX_3090", "num_gpus=1", "cuda_vers>=13",
                  "reliability>=0.95", "inet_down>=300", "inet_up>=100",
                  "cpu_cores>=8", "cpu_ram>=32", "disk_space>=128",
                  "duration>=72"]:
        assert piece in q
    assert "8.0" not in q


def test_build_search_query_omits_zero_gates():
    sel = {**os_.DEFAULTS, "min_reliability": 0, "min_inet_down_mbps": 0,
           "min_inet_up_mbps": 0, "min_cpu_cores": 0, "min_cpu_ram_gb": 0,
           "min_disk_gb": 0, "min_duration_days": 0, "max_price_per_hour": 0}
    q = os_.build_search_query("RTX_3060", "12", sel)
    assert q == "gpu_name=RTX_3060 num_gpus=1 cuda_vers>=12"


def test_build_search_query_caps_price_when_set():
    sel = {**os_.DEFAULTS, "max_price_per_hour": 0.5}
    q = os_.build_search_query("RTX_3090", "13", sel)
    assert "dph_total<=0.5" in q


def test_build_search_query_max_gates_and_price_band():
    sel = {**os_.DEFAULTS, "min_inet_down_mbps": 300, "max_inet_down_mbps": 1000,
           "min_price_per_hour": 0.05, "max_price_per_hour": 0.5}
    q = os_.build_search_query("RTX_3090", "13", sel)
    assert "inet_down>=300" in q and "inet_down<=1000" in q
    assert "dph_total>=0.05" in q and "dph_total<=0.5" in q


def test_max_download_gate_filters_high_speeds():
    sel = {**os_.DEFAULTS, "max_inet_down_mbps": 500}
    assert os_.passes_filters(OFFERS[0], sel, set())       # 500 ≤ 500 → ok
    assert not os_.passes_filters(OFFERS[1], sel, set())   # 900 > 500 → dropped


def test_min_price_gate_filters_cheap_junk():
    sel = {**os_.DEFAULTS, "min_price_per_hour": 0.10}
    cheap = {**OFFERS[0], "id": 99, "dph_total": 0.05}
    assert not os_.passes_filters(cheap, sel, set())
    assert os_.passes_filters(OFFERS[0], sel, set())


def test_geo_block_matches_country_code_in_region_string():
    # vast reports geolocation as "Region, CC" — "US" must block "California, US"
    offer = {**OFFERS[0], "geolocation": "California, US"}
    assert not os_.passes_filters(offer, os_.DEFAULTS, {"US"})
    assert os_.passes_filters(offer, os_.DEFAULTS, {"NL"})      # unrelated block
    assert not os_.passes_filters(offer, os_.DEFAULTS, {"CALIFORNIA, US"})  # exact


def test_strict_gates_drop_bad_offers():
    ranked = os_.rank_offers(OFFERS, STRICT)
    assert [o["id"] for o in ranked] == [1]   # only the all-good offer survives


def test_no_default_gates_rank_everything_by_score():
    ranked = os_.rank_offers(OFFERS, os_.DEFAULTS)
    # order by (reliability, min-inet, cpu, -price): 1 (.99) > 4 (.98) > 3 (.97) > 2 (.80)
    assert [o["id"] for o in ranked] == [1, 4, 3, 2]


def test_rank_offers_blocked_countries():
    ranked = os_.rank_offers(OFFERS, {**os_.DEFAULTS, "blocked_countries": "NL"})
    assert [o["id"] for o in ranked] == [4, 3, 2]
    ranked2 = os_.rank_offers(OFFERS, {**os_.DEFAULTS, "blocked_countries": "us,de"})
    assert [o["id"] for o in ranked2] == [1, 3]


def test_rank_offers_price_cap():
    ranked = os_.rank_offers(OFFERS, {**os_.DEFAULTS, "max_price_per_hour": 0.21})
    assert [o["id"] for o in ranked] == [2]   # only offer 2 (.20) is under the cap


def test_gpu_frac_gate_is_configurable():
    ranked = os_.rank_offers(OFFERS, {**os_.DEFAULTS, "min_gpu_frac": 1.0})
    # fractional offer 4 dropped; the rest keep score order (rel .99 > .97 > .80)
    assert [o["id"] for o in ranked] == [1, 3, 2]


def test_reliability_bucket_beats_raw_reliability_tie():
    # Two hosts both "reliable": internet must decide, not float noise in rel.
    a = {**OFFERS[0], "id": 11, "reliability": 0.9999, "inet_down": 100,
         "inet_up": 100, "dph_total": 0.20}
    b = {**OFFERS[0], "id": 12, "reliability": 0.9997, "inet_down": 2000,
         "inet_up": 2000, "dph_total": 0.25}
    ranked = os_.rank_offers([a, b], os_.DEFAULTS)
    assert [o["id"] for o in ranked] == [12, 11]   # faster internet first


def test_cheapest_mode_orders_by_price():
    ranked = os_.rank_offers(OFFERS, os_.DEFAULTS, mode="cheapest")
    assert [o["id"] for o in ranked] == [2, 4, 1, 3]   # price asc: .20 .22 .25 .30


def test_cheapest_mode_still_respects_gates():
    # strict gates → only offer 1 passes, so cheapest == best here
    ranked = os_.rank_offers(OFFERS, STRICT, mode="cheapest")
    assert [o["id"] for o in ranked] == [1]


def test_price_guard_drops_outlier_from_large_pool():
    base = {**OFFERS[0], "id": 21, "dph_total": 0.10}
    pool = [base, {**base, "id": 22, "dph_total": 0.12},
            {**base, "id": 23, "dph_total": 0.11},
            {**base, "id": 24, "dph_total": 0.09},
            {**base, "id": 25, "dph_total": 1.50}]  # 12× the median → outlier
    ranked = os_.rank_offers(pool, os_.DEFAULTS)
    ids = [o["id"] for o in ranked]
    assert 25 not in ids                      # outlier dropped
    assert len(ids) == 4


def test_price_guard_skipped_on_tiny_pool():
    # < 5 offers → guard is skipped, so an outlier survives the ranking
    base = {**OFFERS[0], "id": 31, "dph_total": 0.10}
    pool = [base, {**base, "id": 32, "dph_total": 0.11},
            {**base, "id": 33, "dph_total": 0.12},
            {**base, "id": 34, "dph_total": 1.00}]
    ids = [o["id"] for o in os_.rank_offers(pool, os_.DEFAULTS)]
    assert 34 in ids


def test_score_offer_fallbacks_for_missing_fields():
    sparse = {"id": 9, "reliability": 0.9, "inet_down": 100,
              "cpu_cores": 8, "dph": 0.15}     # no inet_up / effective / dph_total
    rel, inet, cpu, neg_price = os_.score_offer(sparse)
    assert rel == 0.9
    assert inet == 0.0          # missing inet_up → min is 0
    assert cpu == 8.0           # falls back to cpu_cores
    assert neg_price == -0.15   # falls back to dph


def test_passes_filters_missing_effective_cores_uses_advertised():
    offer = {"id": 5, "reliability": 0.99, "inet_down": 500, "inet_up": 200,
             "cpu_cores": 12, "cpu_ram": 65536, "disk_space": 200,
             "duration": 7 * 86400, "gpu_frac": 1.0, "pcie_bw": 32,
             "geolocation": "FR"}   # cpu_cores_effective absent
    assert os_.passes_filters(offer, {**os_.DEFAULTS, "min_cpu_cores": 8}, set())


def test_garbage_fields_do_not_crash():
    assert os_._num({"inet_down": "fast"}, "inet_down") == 0.0
    assert os_._num({"reliability": None}, "reliability", 0.5) == 0.5


def test_describe_offer_smoke():
    line = os_.describe_offer(OFFERS[0])
    assert "#1" in line and "0.250" in line and "NL" in line
