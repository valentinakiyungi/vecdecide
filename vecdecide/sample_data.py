"""A tiny made-up support-ticket history used by the quickstart and tests.

Each row: (ticket text, team it was routed to, outcome).
Outcome 1.0 = that team resolved it; 0.0 = it had to be re-routed.
"""

TICKETS: list[tuple[str, str, float]] = [
    ("I was charged twice for my subscription this month", "billing", 1.0),
    ("Please refund the duplicate charge on my credit card", "billing", 1.0),
    ("My invoice shows the wrong amount and tax", "billing", 1.0),
    ("How do I update the credit card used for payment", "billing", 1.0),
    ("I need a receipt for my last payment", "billing", 1.0),
    ("Cancel my subscription and refund the annual plan", "billing", 1.0),
    ("The app crashes when I open the settings page", "technical", 1.0),
    ("Getting error 500 when uploading a file", "technical", 1.0),
    ("The export button does nothing after the latest update", "technical", 1.0),
    ("Sync is stuck and my files are not uploading", "technical", 1.0),
    ("The mobile app freezes on the loading screen", "technical", 1.0),
    ("API requests return a timeout error since yesterday", "technical", 1.0),
    ("I forgot my password and the reset email never arrives", "account", 1.0),
    ("I cannot log in, it says my account is locked", "account", 1.0),
    ("How do I change the email address on my account", "account", 1.0),
    ("Please delete my account and all my data", "account", 1.0),
    ("Two factor authentication code is not accepted at login", "account", 1.0),
    ("I want to add a team member to my account", "account", 1.0),
    # Cases that went to the wrong team first and bounced:
    ("I cannot log in to see my invoice", "billing", 0.0),
    ("I cannot log in to download my receipt", "billing", 0.0),
    ("Payment page shows an error 500 when I submit my card", "billing", 0.0),
    ("Payment page crashes when I submit my credit card", "technical", 1.0),
    ("Locked out of my account so I cannot see my invoice", "account", 1.0),
]

SCORED_TICKETS: list[tuple[str, float]] = [
    # (ticket text, hours it took to resolve)
    ("How do I update the credit card used for payment", 1.0),
    ("I need a receipt for my last payment", 0.5),
    ("How do I change the email address on my account", 1.0),
    ("I want to add a team member to my account", 1.5),
    ("I forgot my password and the reset email never arrives", 2.0),
    ("Please refund the duplicate charge on my credit card", 4.0),
    ("I was charged twice for my subscription this month", 5.0),
    ("My invoice shows the wrong amount and tax", 6.0),
    ("The app crashes when I open the settings page", 20.0),
    ("Getting error 500 when uploading a file", 24.0),
    ("Sync is stuck and my files are not uploading", 30.0),
    ("The mobile app freezes on the loading screen", 22.0),
    ("API requests return a timeout error since yesterday", 36.0),
    ("The export button does nothing after the latest update", 18.0),
    ("Payment page crashes when I submit my credit card", 26.0),
    ("I cannot log in, it says my account is locked", 3.0),
]
