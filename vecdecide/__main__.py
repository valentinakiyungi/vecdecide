"""Command line: ``python -m vecdecide quickstart`` or ``... route "text"``."""
from __future__ import annotations

import argparse

from . import Decider, OutcomeStore
from .sample_data import SCORED_TICKETS, TICKETS


def _build() -> tuple[Decider, OutcomeStore]:
    d = Decider()
    store = OutcomeStore(d.embedder.dim)
    for text, team, outcome in TICKETS:
        d.remember(store, text, team, outcome)
    return d, store


def _show(d: Decider, store: OutcomeStore, ticket: str) -> None:
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="vecdecide")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("quickstart", help="route sample tickets, no downloads")
    r = sub.add_parser("route", help="route your own ticket text")
    r.add_argument("text")
    args = parser.parse_args(argv)

    d, store = _build()
    if args.cmd == "route":
        _show(d, store, args.text)
        return

    print(f"Loaded {len(store)} past tickets into an outcome store.")
    for ticket in [
        "I got charged two times, please refund one payment",
        "The app keeps crashing after the update",
        "I cannot log in to check my invoice",
    ]:
        _show(d, store, ticket)

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


if __name__ == "__main__":
    main()
