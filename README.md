# vecdecide

Small, fast decisions over embeddings. Instead of asking a language model to
write an answer, you embed the text once and ask typed questions of the
vector: *which of these options?*, *yes or no?*, *what number?*

It is a minimal, from-scratch take on the idea behind
[venn-vdm](https://github.com/markrussinovich/venn-vdm). It shares no code with
that project and is far smaller: about 1,200 lines of Python on numpy.

## Install

```
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
vecdecide quickstart        # demo on built-in tickets, no files needed
python -m pytest
```

## Use it on your own data

Put your examples in a CSV file with a column of text and a column of labels
(the team, category, tag, whatever you want it to choose). An `outcome` column
is optional: a number where higher is better, such as 1 for "this choice
worked" and 0 for "it had to be redone". Without it, every row counts as good.
Any other columns are kept and shown alongside.

```csv
text,label,outcome
I was charged twice this month,billing,1
The app crashes on launch,technical,1
I cannot log in to see my invoice,billing,0
```

`examples/tickets.csv` is a ready-made file to try the commands on.

```
# 1. How well would it work? Cross-validates on your file; nothing is saved.
vecdecide evaluate examples/tickets.csv

# 2. Build a model folder from the file.
vecdecide build examples/tickets.csv -o my-model

# 3. Ask it things.
vecdecide decide my-model "My card was charged two times"
vecdecide decide my-model --method head "Where is my parcel"
vecdecide decide my-model --json "Where is my parcel"      # for other programs

# 4. Run a whole CSV; adds decision and confidence columns.
vecdecide batch my-model new_tickets.csv -o decided.csv

# 5. Teach it: record what actually happened. The next answer uses it.
vecdecide record my-model "The courier lost my box" shipping --outcome 1

vecdecide info my-model            # what is inside
vecdecide <command> --help         # every option
```

Different column names? Add `--text-col message --label-col category`
(and `--outcome-col` if you have one).

### Two methods

- **store** (default): finds the most similar past cases and picks the label
  that worked best for them. It shows those cases as evidence, it learns
  instantly from `record`, and it uses bad outcomes to steer away from
  choices that failed. It answers `(none of these)` when nothing in the history
  is similar enough (`--min-similarity` sets the bar).
- **head**: a small classifier trained on the good outcomes. Often a bit more
  accurate, gives no evidence, and is refitted after each `record`.

Run `evaluate` to see which works better on your data, and compare both
against the baseline it prints: always guessing the most common label.

### Getting better accuracy

The default embedder matches words, not meaning: "billed" and "charged" look
unrelated to it. On the sample file it gets about 71% (store) and 77% (head)
against a 25% baseline. A semantic model usually does much better:

```
pip install -e ".[embedding]"                              # one-off, large download
vecdecide evaluate examples/tickets.csv --embedder st      # compare first
vecdecide build examples/tickets.csv -o my-model --embedder st
```

`--embedder st:<model name>` picks any
[sentence-transformers](https://www.sbert.net) model. The model folder
remembers which embedder it was built with. More examples per label helps
too: aim for at least 20.

## Python API

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
| `vecdecide/project.py` | `Model` folders, CSV reading, cross-validation |
| `vecdecide/__main__.py` | CLI |
| `examples/tickets.csv` | 84 sample tickets to try the commands on |
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
- The sample ticket data is invented. Accuracy on it says little about
  accuracy on yours; run `evaluate` on your own file.
- Model folders are rewritten in full on every `record`. Fine for thousands of
  cases, slow for millions.

## Ideas for next steps

An HTTP endpoint, an approximate-nearest-neighbour index, score questions from
CSV, and drift detection between two sets of vectors.

MIT licensed.
