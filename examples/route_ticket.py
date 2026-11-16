"""Route a support ticket three ways. Run from the project root:

    python examples/route_ticket.py
"""
from vecdecide import Decider, OutcomeStore
from vecdecide.sample_data import TICKETS

# To use a real semantic model instead of the hashing baseline:
#   from vecdecide import SentenceTransformerEmbedder
#   d = Decider(SentenceTransformerEmbedder())
d = Decider()
ticket = "I was billed twice and want one charge refunded"

# 1. Zero-shot: only option descriptions, no history.
zero = d.choose(ticket, {
    "billing": "payments, charges, refunds, invoices, credit card",
    "technical": "crashes, errors, bugs, sync, upload problems",
    "account": "login, password, email address, account access",
})
print("zero-shot     :", zero.answer, zero.scores)

# 2. Outcome store: what worked for similar past tickets?
store = OutcomeStore(d.embedder.dim)
for text, team, outcome in TICKETS:
    d.remember(store, text, team, outcome)
stored = d.decide_from_outcomes(ticket, store)
print("outcome store :", stored.answer, f"(support {stored.confidence:.0%})")
for n in stored.evidence[:3]:
    print(f"   {n.similarity:.2f} {n.option:<9} outcome={n.outcome} | {n.text}")

# 3. Compile the store into a small head: no neighbour search at decision time.
head = d.compile_store(store)
print("compiled head :", d.choose_with_head(ticket, head, none_below=0.4).answer,
      f"({head.n_params:,} parameters)")

# Close the loop: after the ticket is handled, record how it went.
d.remember(store, ticket, stored.answer, outcome=1.0)
