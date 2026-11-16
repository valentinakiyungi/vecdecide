"""vecdecide: small, fast decisions over embeddings."""
from .decider import Decider, Decision
from .embedders import Embedder, HashingEmbedder, SentenceTransformerEmbedder, normalize
from .heads import ChoiceHead, ScoreHead, ScorePrediction
from .store import Neighbor, OutcomeStore

__version__ = "0.1.0"
__all__ = [
    "Decider", "Decision", "Embedder", "HashingEmbedder",
    "SentenceTransformerEmbedder", "normalize", "ChoiceHead", "ScoreHead",
    "ScorePrediction", "Neighbor", "OutcomeStore",
]
