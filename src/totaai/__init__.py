"""TotaAI: proste AI z polskim API."""

from .backend import cuda_dostepna
from .model import Model
from .narzedzia import blad_sredni_bezwzgledny, dokladnosc, podziel_dane
from .optymalizatory import SGD, Adam
from .straty import MAE, MSE, EntropiaBinarna, EntropiaKrzyzowa
from .tensor import Tensor
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

__version__ = "0.4.0"

__all__ = [
    "MAE",
    "MSE",
    "SGD",
    "Adam",
    "Dropout",
    "EntropiaBinarna",
    "EntropiaKrzyzowa",
    "LeakyReLU",
    "Model",
    "ReLU",
    "Sigmoid",
    "Softmax",
    "Tanh",
    "Tensor",
    "Warstwa",
    "WarstwaLiniowa",
    "blad_sredni_bezwzgledny",
    "cuda_dostepna",
    "dokladnosc",
    "podziel_dane",
]
