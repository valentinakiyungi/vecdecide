import numpy as np
import pytest

from vecdecide import ChoiceHead, Decider, HashingEmbedder, OutcomeStore, ScoreHead
from vecdecide.sample_data import SCORED_TICKETS, TICKETS


@pytest.fixture
def setup():
    d = Decider()
    store = OutcomeStore(d.embedder.dim)
    for text, team, outcome in TICKETS:
        d.remember(store, text, team, outcome)
    return d, store


def test_embedder_is_deterministic_and_normalised():
    e = HashingEmbedder(dim=256)
    a = e.embed(["hello world", "refund my payment"])
    b = e.embed(["hello world", "refund my payment"])
    assert a.shape == (2, 256)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(np.linalg.norm(a, axis=1), 1.0, rtol=1e-5)
    assert np.linalg.norm(e.embed([""])[0]) == 0.0


def test_similar_text_is_closer():
    e = HashingEmbedder()
    q, near, far = e.embed(["refund my payment", "please refund the payment", "app crashes on launch"])
    assert q @ near > q @ far


def test_zero_shot_choice_and_none():
    d = Decider()
    opts = {"billing": "payment refund invoice charge", "technical": "crash error bug freeze"}
    assert d.choose("I want a refund for this charge", opts).answer == "billing"
    assert d.choose("zzz qqq", opts, none_below=0.1).answer is None


def test_yes_no():
    d = Decider()
    r = d.yes_no("the app crashes with an error", yes="crash error bug", no="payment refund invoice")
    assert r.answer is True


def test_outcome_store_routes(setup):
    d, store = setup
    assert d.decide_from_outcomes("please refund the double charge", store).answer == "billing"
    assert d.decide_from_outcomes("the app crashes after the update", store).answer == "technical"


def test_bad_outcomes_steer_away(setup):
    d, store = setup
    # Nearest cases went to billing and bounced; the store should avoid billing.
    r = d.decide_from_outcomes("I cannot log in to check my invoice", store, k=5)
    assert r.evidence[0].option == "billing" and r.evidence[0].outcome == 0.0
    assert r.answer == "account"


def test_no_similar_case_returns_none(setup):
    d, store = setup
    assert d.decide_from_outcomes("zzzz qqqq xxxx", store, min_similarity=0.3).answer is None
    assert d.decide_from_outcomes("anything", OutcomeStore(d.embedder.dim)).answer is None


def test_write_back_changes_answer(setup):
    d, store = setup
    q = "Where can I find the VAT number on my bill"
    assert d.decide_from_outcomes(q, store, min_similarity=0.3).answer is None
    d.remember(store, "What VAT number appears on my bill", "billing", 1.0)
    assert d.decide_from_outcomes(q, store, min_similarity=0.3).answer == "billing"


def test_store_save_load(setup, tmp_path):
    d, store = setup
    store.save(tmp_path / "s")
    loaded = OutcomeStore.load(tmp_path / "s")
    assert len(loaded) == len(store)
    q = d.embed("refund my charge")
    assert [n.index for n in loaded.nearest(q, 3)] == [n.index for n in store.nearest(q, 3)]


def test_store_rejects_wrong_dim():
    with pytest.raises(ValueError):
        OutcomeStore(8).add(np.ones(4), "a")


def test_choice_head_learns_and_abstains(tmp_path):
    d = Decider()
    good = [(t, team) for t, team, o in TICKETS if o >= 0.5]
    head = d.train_choice_head([t for t, _ in good], [l for _, l in good])
    preds = [d.choose_with_head(t, head).answer for t, _ in good]
    assert np.mean([p == l for p, (_, l) in zip(preds, good)]) == 1.0
    assert d.choose_with_head("refund the charge on my card", head).answer == "billing"
    # Softmax confidence is a weak "unfamiliar" signal: gibberish can still score
    # ~0.55 here, so the threshold has to be chosen on real data.
    assert d.choose_with_head("zzzz qqqq", head, none_below=0.7).answer is None
    head.save(tmp_path / "h.npz")
    again = ChoiceHead.load(tmp_path / "h.npz")
    np.testing.assert_allclose(again.predict_proba(d.embed("refund")), head.predict_proba(d.embed("refund")))


def test_compiled_head_matches_store(setup):
    d, store = setup
    head = d.compile_store(store)
    assert d.choose_with_head("I cannot log in to check my invoice", head).answer == "account"


def test_score_head_interval():
    d = Decider()
    head = d.train_score_head([t for t, _ in SCORED_TICKETS], [h for _, h in SCORED_TICKETS])
    slow = d.score("the app crashes with an error on upload", head)
    fast = d.score("I need a receipt for my payment", head)
    assert slow.value > fast.value
    assert slow.low < slow.value < slow.high


def test_score_head_coverage_on_synthetic_data():
    rng = np.random.default_rng(1)
    w = rng.normal(size=16)
    X = rng.normal(size=(600, 16))
    y = X @ w + rng.normal(scale=0.5, size=600)
    head = ScoreHead.fit(X[:400], y[:400], l2=1e-3)
    inside = [head.predict(x).low <= t <= head.predict(x).high for x, t in zip(X[400:], y[400:])]
    assert 0.7 < np.mean(inside) < 0.9
