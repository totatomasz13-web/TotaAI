from __future__ import annotations
import numpy as np
from .tensor import Tensor


class Warstwa:
    def parametry(self): return []
    def __call__(self, x): return self.przepusc(x)
    def przepusc(self, x): raise NotImplementedError


class WarstwaLiniowa(Warstwa):
    def __init__(self, wejscia, wyjscia):
        skala = np.sqrt(2 / wejscia)
        self.wagi = Tensor(np.random.default_rng().normal(0, skala, (wejscia, wyjscia)), True)
        self.bias = Tensor(np.zeros(wyjscia), True)
    def przepusc(self, x): return x @ self.wagi + self.bias
    def parametry(self): return [self.wagi, self.bias]


class ReLU(Warstwa):
    def przepusc(self, x):
        wynik = Tensor(np.maximum(0, x.dane), x.wymaga_gradientu, (x,))
        wynik._wstecz = lambda: x._dodaj_gradient(wynik.gradient * (x.dane > 0)) if wynik.gradient is not None and x.wymaga_gradientu else None
        return wynik


class Sigmoid(Warstwa):
    def przepusc(self, x):
        dane = 1 / (1 + np.exp(-np.clip(x.dane, -20, 20)))
        wynik = Tensor(dane, x.wymaga_gradientu, (x,))
        wynik._wstecz = lambda: x._dodaj_gradient(wynik.gradient * dane * (1 - dane)) if wynik.gradient is not None and x.wymaga_gradientu else None
        return wynik


class Softmax(Warstwa):
    def przepusc(self, x):
        wykladniki = np.exp(x.dane - np.max(x.dane, axis=-1, keepdims=True))
        dane = wykladniki / wykladniki.sum(axis=-1, keepdims=True)
        wynik = Tensor(dane, x.wymaga_gradientu, (x,))
        def wstecz():
            if wynik.gradient is not None and x.wymaga_gradientu:
                x._dodaj_gradient(dane * (wynik.gradient - (wynik.gradient * dane).sum(axis=-1, keepdims=True)))
        wynik._wstecz = wstecz
        return wynik
