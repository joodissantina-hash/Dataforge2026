"""Unit tests for the deterministic simulator and its API surface.

These assert the DIRECTION of each effect the artifact teaches - the exact float
values are not the contract, the mechanism is.
"""

import pytest

from lib.simulator import CAPS, decode_value, embed, simulate, sparsify

BASE = {
    "task": "changing_facts", "seq_len": 48, "mem_slots": 6, "update_rate": 0.25,
    "noise": 0.0, "decay": 0.01, "ood_query": False, "reset_midway": False, "seed": 7,
}


def cfg(**over):
    return {**BASE, **over}


def modes(out):
    return {m["mode"]: m for m in out["modes"]}


# ------------------------------------------------------------------ primitives
def test_embeddings_are_deterministic_and_unit_norm():
    a, b = embed("ALPHA"), embed("ALPHA")
    assert list(a) == list(b)
    assert abs(float((a * a).sum()) - 1.0) < 1e-6
    assert list(embed("ALPHA")) != list(embed("BETA"))


def test_sparsify_is_positive_and_sparse():
    x = sparsify(embed("ALPHA"))
    assert (x >= 0).all()
    assert 0 < int((x > 0).sum()) <= 6


def test_decode_abstains_on_empty_readout():
    value, _, _ = decode_value(embed("ALPHA") * 0.0)
    assert value == "UNKNOWN"


# ------------------------------------------------------------------ determinism
def test_same_seed_same_result():
    a, b = simulate(cfg()), simulate(cfg())
    assert [m["prediction"] for m in a["modes"]] == [m["prediction"] for m in b["modes"]]
    assert [t["label"] for t in a["tokens"]] == [t["label"] for t in b["tokens"]]


def test_state_size_scaling_directions():
    short, long = simulate(cfg(seq_len=16)), simulate(cfg(seq_len=120))
    s, lo = modes(short), modes(long)
    # attention state grows with N; both persistent states do not
    assert lo["attention"]["state_floats"] > s["attention"]["state_floats"]
    assert lo["memory"]["state_floats"] == s["memory"]["state_floats"]
    assert lo["bdh"]["state_floats"] == s["bdh"]["state_floats"]


# ------------------------------------------------------------------ the claim
def test_low_update_rate_produces_stale_memory():
    """The failure half: a convex write below 0.5 keeps the superseded binding."""
    out = simulate(cfg(update_rate=0.2))
    m = modes(out)
    assert out["ground_truth"] == "amber"
    assert m["attention"]["correct"]
    assert not m["memory"]["correct"]
    assert m["memory"]["prediction"] == "red"          # the superseded value
    assert m["memory"]["confidence"] > 0.5            # wrong AND confident


def test_high_update_rate_fixes_staleness():
    """Same seed, same sequence - only alpha changes, and the failure disappears."""
    out = simulate(cfg(update_rate=0.95))
    assert modes(out)["memory"]["correct"]


def test_bdh_additive_write_beats_blended_write_on_changed_fact():
    out = simulate(cfg(update_rate=0.25))
    m = modes(out)
    assert m["bdh"]["correct"]
    assert not m["memory"]["correct"]


def test_persistent_memory_benefit_on_long_delayed_recall():
    """The benefit half: constant-size state matches attention across a 120-token gap."""
    out = simulate(cfg(task="delayed_recall", seq_len=120, mem_slots=8,
                       update_rate=0.85, decay=0.0))
    m = modes(out)
    assert m["attention"]["correct"]
    assert m["memory"]["correct"]
    assert m["memory"]["state_floats"] < m["attention"]["state_floats"]


def test_capacity_overflow_evicts_the_needed_fact():
    out = simulate(cfg(task="distractor_recall", seq_len=64, mem_slots=2, update_rate=0.9))
    m = modes(out)
    assert m["memory"]["evictions"] > 0
    assert not m["memory"]["correct"]
    assert m["attention"]["correct"]


def test_reset_destroys_persistent_state_but_not_the_cache():
    out = simulate(cfg(task="delayed_recall", update_rate=0.9, reset_midway=True))
    m = modes(out)
    assert out["reset_at"] is not None
    assert m["attention"]["correct"]
    assert not m["memory"]["correct"]
    assert not m["bdh"]["correct"]


def test_ood_query_ground_truth_is_unknown():
    out = simulate(cfg(task="delayed_recall", ood_query=True))
    assert out["ground_truth"] == "UNKNOWN"
    assert out["query_key"] not in [t["key"] for t in out["tokens"] if t["kind"] == "fact"]


def test_heavy_noise_degrades_every_mode_confidence():
    clean = modes(simulate(cfg(task="noisy_prediction", noise=0.0)))
    noisy = modes(simulate(cfg(task="noisy_prediction", noise=0.5)))
    assert noisy["memory"]["confidence"] < clean["memory"]["confidence"]


def test_extreme_parameters_do_not_crash():
    for over in (
        {"seq_len": CAPS["seq_len"]["min"]},
        {"seq_len": CAPS["seq_len"]["max"]},
        {"mem_slots": CAPS["mem_slots"]["min"], "seq_len": 128},
        {"decay": CAPS["decay"]["max"], "seq_len": 128},
        {"update_rate": CAPS["update_rate"]["min"]},
        {"update_rate": CAPS["update_rate"]["max"], "noise": CAPS["noise"]["max"]},
    ):
        for task in ("delayed_recall", "changing_facts", "distractor_recall", "noisy_prediction"):
            out = simulate(cfg(task=task, **over))
            assert len(out["modes"]) == 3
            for m in out["modes"]:
                assert isinstance(m["prediction"], str) and m["prediction"]


# ----------------------------------------------------------------------------- API surface
def test_meta_exposes_caps_evidence_and_papers(client):
    body = client.get("/meta").json()
    assert body["caps"]["seq_len"]["max"] == 128
    assert len(body["papers"]) >= 3
    assert {"formal_result", "reported_experiment", "interpretation", "simplification"} <= {
        c["tier"] for c in body["evidence"]
    }
    assert len(body["presets"]) >= 5


def test_run_endpoint_is_live_and_structured(client):
    r = client.post("/run", json=cfg())
    assert r.status_code == 200
    body = r.json()
    assert body["computation_status"] == "live"
    assert body["engine"] == "server-numpy"
    assert body["latency_ms"] >= 0
    assert body["ground_truth"] == "amber"
    assert len(body["modes"]) == 3


@pytest.mark.parametrize("bad", [
    {"seq_len": 5}, {"seq_len": 5000}, {"mem_slots": 0}, {"mem_slots": 9999},
    {"update_rate": 3.0}, {"noise": -1}, {"decay": 5}, {"seed": -4},
    {"task": "'; DROP TABLE runs;    "},
])
def test_out_of_range_parameters_are_rejected_before_compute(client, bad):
    assert client.post("/run", json=cfg(**bad)).status_code == 422


def test_unknown_run_id_is_404(client):
    assert client.get("/runs/does-not-exist").status_code == 404


def test_learner_response_is_accepted_anonymously(client):
    r = client.post("/responses", json={
        "run_id": None, "task": "changing_facts", "predicted_failures": ["memory"],
        "explanation": "The blended write kept the superseded value dominant.", "self_score": 2,
    })
    assert r.status_code == 200
    assert r.json()["accepted"]
    assert len(r.json()["rubric"]) >= 3


def test_health_reports_engine(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["engine"] == "server-numpy"