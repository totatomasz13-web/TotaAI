"""TotaAI: proste AI z polskim API."""

from .tensor import Tensor
from .warstwy import Warstwa, WarstwaLiniowa, ReLU, Sigmoid, Softmax
from .straty import MSE, EntropiaKrzyzowa
from .optymalizatory import SGD, Adam
from .model import Model

__version__ = "0.2.0"

__all__ = [
    "Tensor", "Warstwa", "WarstwaLiniowa", "ReLU", "Sigmoid", "Softmax",
    "MSE", "EntropiaKrzyzowa", "SGD", "Adam", "Model",
]
