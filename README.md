# vecdecide

Small, fast decisions over embeddings. Instead of asking a language model to
write an answer, you embed the text once and ask typed questions of the
vector: *which of these options?*, *yes or no?*, *what number?*

It is a minimal, from-scratch take on the idea behind
[venn-vdm](https://github.com/markrussinovich/venn-vdm). It shares no code with
that project and is far smaller: about 500 lines of Python on numpy.

## Install and try

```
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
vecdecide quickstart            # or: python -m vecdecide quickstart
vecdecide route "I can't reset my password"
pytest
```

The quickstart needs no model download. It uses a hashing embedder that only
sees word overlap. For real use, swap in a semantic model:

```
pip install -e ".[embedding]"
```
```python
from vecdecide import Decider, SentenceTransformerEmbedder
d = Decider(SentenceTransformerEmbedder())   # downloads a ~90 MB model once
```

## Three ways to decide

```python
from vecdecide import Decider, OutcomeStore
d = Decider()
```

**1. Zero-shot.** No data. Compares the text to each option's description.

```python
d.choose("I want a refund", {"billing": "payments, refunds", "technical": "crashes, bugs"})
d.yes_no("the app crashed", yes="software failure", no="everything else")
```

**2. Outcome store.** Record past cases, what was chosen and how it went.
A new case gets the option that worked best for its nearest neighbours.
Nothing is trained; writing an outcome back changes the next answer.

```python
store = OutcomeStore(d.embedder.dim)
d.remember(store, "charged twice this month", "billing", outcome=1.0)
d.remember(store, "can't log in to see invoice", "billing", outcome=0.0)  # bounced
decision = d.decide_from_outcomes("double charge on my card", store)
decision.answer, decision.confidence, decision.evidence
```

**3. Trained heads.** A few thousand parameters on top of the frozen embedder.

```python
head = d.train_choice_head(texts, labels)             # classifier
d.choose_with_head(text, head, none_below=0.4)        # None = "none of these"

head = d.compile_store(store)                         # distil a store into a head

score = d.train_score_head(texts, numbers)            # regression
p = d.score(text, score); p.value, p.low, p.high      # with an 80% interval
```

Every call returns a `Decision` (`answer`, `confidence`, `scores`, `method`,
`evidence`) or a `ScorePrediction`. All of them accept a precomputed vector
in place of text, so vectors you already have can be reused for new questions.

## Layout

| File | What it holds |
| --- | --- |
| `vecdecide/embedders.py` | `HashingEmbedder` (baseline), `SentenceTransformerEmbedder` |
| `vecdecide/store.py` | `OutcomeStore`: add, nearest, save, load |
| `vecdecide/heads.py` | `ChoiceHead` (softmax regression), `ScoreHead` (ridge + conformal interval) |
| `vecdecide/decider.py` | `Decider`: the question API |
| `vecdecide/__main__.py` | CLI |
| `examples/route_ticket.py` | The three methods side by side |

## Limits

- The default hashing embedder matches words, not meaning. "Billed" and
  "charged" look unrelated to it. Use a real embedding model for anything real.
- `confidence` values are not calibrated probabilities. Treat them as rankings
  and pick thresholds on your own data.
- `None` ("none of these") is a hint that the case looks unfamiliar, not a
  guarantee.
- Nearest-neighbour search is brute force. Fine to roughly 100k records;
  beyond that, put an index such as FAISS behind `OutcomeStore.nearest`.
- The sample ticket data is invented and tiny. It demonstrates the mechanics,
  not accuracy.

## Ideas for next steps

An HTTP endpoint, an approximate-nearest-neighbour index, a head that outputs
an explicit "none" class, drift detection between two sets of vectors, and
running one question over every row of a stored collection.

MIT licensed.
