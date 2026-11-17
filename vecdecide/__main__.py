"""Command line interface.

    vecdecide quickstart                       demo on built-in sample tickets
    vecdecide build data.csv -o my-model       build a model folder from a CSV
    vecdecide info my-model                    what is in a model folder
    vecdecide decide my-model "some text"      ask for a decision
    vecdecide record my-model "text" LABEL     write a new outcome back
    vecdecide batch my-model in.csv -o out.csv decide every row of a CSV
    vecdecide evaluate data.csv                hold-out accuracy on your data
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from . import Decider, OutcomeStore
from .project import DEFAULT_EMBEDDER, Model, evaluate, read_rows, read_texts
from .sample_data import SCORED_TICKETS, TICKETS


# -- quickstart (unchanged demo) ----------------------------------------------
def _demo_show(d: Decider, store: OutcomeStore, ticket: str) -> None:
    print(f'\nTicket: "{ticket}"')
    res = d.decide_from_outcomes(ticket, store, k=5)
    print(f"  From past outcomes -> {res.answer}  (support {res.confidence:.0%})")
    for n in res.evidence[:3]:
        mark = "resolved" if n.outcome >= 0.5 else "bounced"
        print(f"      {n.similarity:.2f}  {n.option:<9} {mark:<8}  {n.text}")
    head = d.compile_store(store)
    res2 = d.choose_with_head(ticket, head, none_below=0.4)
    print(f"  Compiled head      -> {res2.answer}  (p={res2.confidence:.2f}, "
          f"{head.n_params:,} parameters)")


def cmd_quickstart(args) -> None:
    d = Decider()
    store = OutcomeStore(d.embedder.dim)
    for text, team, outcome in TICKETS:
        d.remember(store, text, team, outcome)
    if args.text:
        _demo_show(d, store, args.text)
        return
    print(f"Loaded {len(store)} past tickets into an outcome store.")
    for ticket in [
        "I got charged two times, please refund one payment",
        "The app keeps crashing after the update",
        "I cannot log in to check my invoice",
    ]:
        _demo_show(d, store, ticket)

    print("\nScore question: how many hours will this take?")
    head = d.train_score_head([t for t, _ in SCORED_TICKETS], [h for _, h in SCORED_TICKETS])
    for ticket in ["The app crashes on upload with an error", "I need a receipt for my payment"]:
        p = d.score(ticket, head)
        print(f'  "{ticket}" -> {p.value:.1f}h  (80% interval {max(p.low, 0):.1f} to {p.high:.1f})')

    print("\nWrite-back: record a new outcome and the next answer changes.")
    q = "Where can I find the VAT number on my bill"
    print(f'  before: "{q}" -> {d.decide_from_outcomes(q, store, k=5, min_similarity=0.3).answer}')
    d.remember(store, "What VAT number appears on my bill", "billing", 1.0)
    print(f'  after : "{q}" -> {d.decide_from_outcomes(q, store, k=5, min_similarity=0.3).answer}')
    print("\nNext: try it on your own data with  vecdecide build yourfile.csv -o my-model")


# -- working with your own data ------------------------------------------------
def cmd_build(args) -> None:
    rows = read_rows(args.csv, args.text_col, args.label_col, args.outcome_col)
    model = Model.build(rows, args.embedder, train_head=not args.no_head,
                        min_outcome=args.min_outcome)
    out = model.save(args.output)
    counts: dict[str, int] = {}
    for r in rows:
        counts[r.label] = counts.get(r.label, 0) + 1
    print(f"Built {out} from {len(rows)} rows of {Path(args.csv).name} "
          f"(embedder: {args.embedder}).")
    for label, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {label:<20} {n}")
    if model.head is not None:
        print(f"Trained a head with {model.head.n_params:,} parameters.")
    print(f'\nAsk it something:  vecdecide decide {out} "your text here"')


def cmd_info(args) -> None:
    model = Model.load(args.model)
    cfg = model.config
    print(f"Model folder : {args.model}")
    print(f"Embedder     : {cfg['embedder']} ({cfg['dim']} dimensions)")
    print(f"Cases        : {len(model.store)}")
    print(f"Head         : {'yes, ' + format(model.head.n_params, ',') + ' parameters' if model.head else 'no'}")
    print("Labels       :")
    good, bad = {}, {}
    for o, v in zip(model.store.options, model.store.outcomes):
        d = good if v >= cfg.get("min_outcome", 0.5) else bad
        d[o] = d.get(o, 0) + 1
    for label in model.labels:
        print(f"  {label:<20} {good.get(label, 0)} good, {bad.get(label, 0)} bad")


def _decision_dict(text: str, dec, top: int = 3) -> dict:
    out = {
        "text": text,
        "answer": dec.answer,
        "confidence": round(dec.confidence, 4),
        "method": dec.method,
        "scores": {k: round(v, 4) for k, v in dec.scores.items()},
    }
    if dec.evidence:
        out["evidence"] = [
            {"similarity": round(n.similarity, 4), "label": n.option,
             "outcome": n.outcome, "text": n.text}
            for n in dec.evidence[:top]
        ]
    return out


def cmd_decide(args) -> None:
    model = Model.load(args.model)
    texts = args.text or [line.strip() for line in sys.stdin if line.strip()]
    if not texts:
        sys.exit("give some text, e.g.  vecdecide decide my-model \"my text\"")
    for text in texts:
        dec = model.decide(text, args.method, k=args.k, min_similarity=args.min_similarity,
                           none_below=args.none_below)
        if args.json:
            print(json.dumps(_decision_dict(text, dec, args.show)))
            continue
        answer = dec.answer if dec.answer is not None else "(none of these)"
        print(f'"{text}"\n  -> {answer}   confidence {dec.confidence:.0%}  [{dec.method}]')
        ranked = sorted(dec.scores.items(), key=lambda x: -x[1])
        print("     " + "  ".join(f"{k}={v:.2f}" for k, v in ranked[:5]))
        for n in dec.evidence[: args.show]:
            mark = "good" if n.outcome >= model.config.get("min_outcome", 0.5) else "bad "
            print(f"     {n.similarity:.2f}  {n.option:<14} {mark}  {(n.text or '')[:70]}")


def cmd_record(args) -> None:
    model = Model.load(args.model)
    model.record(args.text, args.label, args.outcome)
    if not args.no_retrain:
        model.retrain()
    model.save(args.model)
    print(f"Recorded: \"{args.text}\" -> {args.label} (outcome {args.outcome}). "
          f"Model now has {len(model.store)} cases.")


def cmd_batch(args) -> None:
    model = Model.load(args.model)
    records, texts = read_texts(args.csv, args.text_col)
    cols = list(records[0].keys()) if records else [args.text_col]
    extra = ["decision", "confidence", "method"]
    out_f = open(args.output, "w", newline="", encoding="utf-8") if args.output else sys.stdout
    try:
        w = csv.DictWriter(out_f, fieldnames=cols + [c for c in extra if c not in cols])
        w.writeheader()
        for rec, text in zip(records, texts):
            if text:
                dec = model.decide(text, args.method, k=args.k,
                                   min_similarity=args.min_similarity, none_below=args.none_below)
                rec.update(decision="" if dec.answer is None else dec.answer,
                           confidence=f"{dec.confidence:.4f}", method=dec.method)
            else:
                rec.update(decision="", confidence="", method="")
            w.writerow(rec)
    finally:
        if args.output:
            out_f.close()
    if args.output:
        print(f"Wrote {len(records)} decisions to {args.output}")


def cmd_evaluate(args) -> None:
    rows = read_rows(args.csv, args.text_col, args.label_col, args.outcome_col)
    results, n_test = evaluate(rows, args.embedder, folds=args.folds, seed=args.seed, k=args.k,
                               min_outcome=args.min_outcome, min_similarity=args.min_similarity)
    print(f"{args.folds}-fold cross-validation on {len(rows)} rows: each of {n_test} good-outcome "
          f"rows was\npredicted by a model built without it (embedder: {args.embedder}).\n")
    for r in results:
        print(f"{r.method:<32} accuracy {r.accuracy:6.1%}   "
              f"({round(r.accuracy * r.total)}/{r.total}, answered {r.answered})")
    for r in results:
        if not r.per_label:
            continue
        print(f"\nPer label, {r.method}:")
        for label, (ok, n) in sorted(r.per_label.items()):
            wrong = {p: c for (t, p), c in r.confusion.items() if t == label and p != label}
            note = ", ".join(f"{c} as {p}" for p, c in sorted(wrong.items(), key=lambda x: -x[1]))
            print(f"  {label:<20} {ok}/{n}" + (f"   mistaken: {note}" if note else ""))
    if args.embedder == DEFAULT_EMBEDDER:
        print("\nTip: the default embedder only matches words. For meaning, install the "
              "extra\n     (pip install -e \".[embedding]\") and add  --embedder st")


# -- argument parsing ------------------------------------------------------------
def _csv_cols(p, label=True, outcome=True):
    p.add_argument("--text-col", default="text", help="column holding the text (default: text)")
    if label:
        p.add_argument("--label-col", default="label",
                       help="column holding the label/team/category (default: label)")
    if outcome:
        p.add_argument("--outcome-col", default="outcome",
                       help="optional column with how well it went, higher is better "
                            "(default: outcome; if missing, every row counts as good)")


def _decide_opts(p):
    p.add_argument("--method", choices=["store", "head"], default="store",
                   help="store: nearest past outcomes (default); head: trained classifier")
    p.add_argument("-k", type=int, default=8, help="past cases to consult (store method)")
    p.add_argument("--min-similarity", type=float, default=None,
                   help="answer 'none' if no past case is this similar (store method; "
                        "default 0.10 for hashing, 0.35 for st)")
    p.add_argument("--none-below", type=float, default=None,
                   help="answer 'none' if the head's top probability is below this (head method)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vecdecide",
        description="Small, fast decisions over embeddings, from your own CSV data.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("quickstart", help="demo on built-in sample tickets, no files needed")
    p.add_argument("text", nargs="?", help="route this text against the sample tickets")
    p.set_defaults(func=cmd_quickstart)

    # kept for compatibility with the first version
    p = sub.add_parser("route", help="route text against the built-in sample tickets")
    p.add_argument("text")
    p.set_defaults(func=cmd_quickstart)

    p = sub.add_parser("build", help="build a model folder from a CSV of labelled examples")
    p.add_argument("csv")
    p.add_argument("-o", "--output", required=True, help="model folder to create")
    p.add_argument("--embedder", default=DEFAULT_EMBEDDER,
                   help="hashing (default, no download) or st / st:<model> for a semantic model")
    p.add_argument("--min-outcome", type=float, default=0.5,
                   help="outcomes at or above this count as good (default 0.5)")
    p.add_argument("--no-head", action="store_true", help="skip training the head")
    _csv_cols(p)
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("info", help="show what is in a model folder")
    p.add_argument("model")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("decide", help="decide for one or more texts (or lines on stdin)")
    p.add_argument("model")
    p.add_argument("text", nargs="*")
    p.add_argument("--json", action="store_true", help="one JSON object per line")
    p.add_argument("--show", type=int, default=3, help="past cases to show as evidence")
    _decide_opts(p)
    p.set_defaults(func=cmd_decide)

    p = sub.add_parser("record", help="write a new case and its outcome into a model")
    p.add_argument("model")
    p.add_argument("text")
    p.add_argument("label")
    p.add_argument("--outcome", type=float, default=1.0,
                   help="how well it went, higher is better (default 1.0)")
    p.add_argument("--no-retrain", action="store_true", help="don't refit the head afterwards")
    p.set_defaults(func=cmd_record)

    p = sub.add_parser("batch", help="decide for every row of a CSV and write the results")
    p.add_argument("model")
    p.add_argument("csv")
    p.add_argument("-o", "--output", help="output CSV (default: print to screen)")
    _csv_cols(p, label=False, outcome=False)
    _decide_opts(p)
    p.set_defaults(func=cmd_batch)

    p = sub.add_parser("evaluate", help="measure accuracy on held-out rows of your CSV")
    p.add_argument("csv")
    p.add_argument("--embedder", default=DEFAULT_EMBEDDER)
    p.add_argument("--folds", type=int, default=5, help="cross-validation folds (default 5)")
    p.add_argument("--min-similarity", type=float, default=None,
                   help="store method: answer 'none' below this similarity")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("-k", type=int, default=8)
    p.add_argument("--min-outcome", type=float, default=0.5)
    _csv_cols(p)
    p.set_defaults(func=cmd_evaluate)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except (ValueError, FileNotFoundError, ImportError) as exc:
        sys.exit(f"error: {exc}")


if __name__ == "__main__":
    main()
