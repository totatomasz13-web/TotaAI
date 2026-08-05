"""TotaAI: proste AI z polskim API."""

from .backend import cuda_dostepna
from .model import Model
from .narzedzia import blad_sredni_bezwzgledny, dokladnosc, podziel_dane
from .optymalizatory import SGD, Adam
from .straty import MAE, MSE, EntropiaBinarna, EntropiaKrzyzowa
from .tensor import Tensor
from .transformer import (
    AdamW,
    CrossEntropyLoss,
    Embedding,
    FeedForward,
    LayerNorm,
    Linear,
    MultiHeadAttention,
    PositionalEncoding,
    SelfAttention,
    Tokenizer,
    TransformerBlock,
    TransformerLM,
)
from .warstwy import (
    Dropout,
    LeakyReLU,
    ReLU,
    Sigmoid,
    Softmax,
    Tanh,
    Warstwa,
    WarstwaLiniowa,
)

__version__ = "1.0.0"

__all__ = [
    "MAE",
    "MSE",
    "SGD",
    "Adam",
    "AdamW",
    "CrossEntropyLoss",
    "Dropout",
    "Embedding",
    "EntropiaBinarna",
    "EntropiaKrzyzowa",
    "FeedForward",
    "LeakyReLU",
    "LayerNorm",
    "Linear",
    "Model",
    "MultiHeadAttention",
    "PositionalEncoding",
    "ReLU",
    "Sigmoid",
    "Softmax",
    "SelfAttention",
    "Tanh",
    "Tensor",
    "Tokenizer",
    "Warstwa",
    "WarstwaLiniowa",
    "TransformerBlock",
    "TransformerLM",
    "blad_sredni_bezwzgledny",
    "cuda_dostepna",
    "dokladnosc",
    "podziel_dane",
]
