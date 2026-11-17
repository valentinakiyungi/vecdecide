"""vecdecide: small, fast decisions over embeddings."""
from .decider import Decider, Decision
from .embedders import Embedder, HashingEmbedder, SentenceTransformerEmbedder, normalize
from .heads import ChoiceHead, ScoreHead, ScorePrediction
from .store import Neighbor, OutcomeStore
from .project import Model, Row, evaluate, make_embedder, read_rows

__version__ = "0.2.0"
__all__ = [
    "Decider", "Decision", "Embedder", "HashingEmbedder",
    "SentenceTransformerEmbedder", "normalize", "ChoiceHead", "ScoreHead",
    "ScorePrediction", "Neighbor", "OutcomeStore",
    "Model", "Row", "evaluate", "make_embedder", "read_rows",
]
