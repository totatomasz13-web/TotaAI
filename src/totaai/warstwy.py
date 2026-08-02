from __future__ import annotations

import numpy as np

from .tensor import Tensor


class Warstwa:
    """Bazowa klasa warstwy. Nadpisz `przepusc()` dla własnej implementacji."""

    def __init__(self):
        self.szkolenie = True

    def parametry(self):
        return []

    def ustaw_tryb(self, szkolenie: bool):
        self.szkolenie = szkolenie

    def __call__(self, x):
        return self.przepusc(x)

    def przepusc(self, x):
        raise NotImplementedError


class WarstwaLiniowa(Warstwa):
    """Warstwa gęsta wykonująca `x @ wagi + bias`."""

    def __init__(self, wejscia, wyjscia):
        super().__init__()
        skala = np.sqrt(2 / wejscia)
        self.wagi = Tensor(np.random.default_rng().normal(0, skala, (wejscia, wyjscia)), True)
        self.bias = Tensor(np.zeros(wyjscia), True)

    def przepusc(self, x):
        return x @ self.wagi + self.bias

    def parametry(self):
        return [self.wagi, self.bias]


class ReLU(Warstwa):
    def __init__(self):
        super().__init__()

    def przepusc(self, x):
        wynik = Tensor(np.maximum(0, x.dane), x.wymaga_gradientu, (x,))
        wynik._wstecz = lambda: x._dodaj_gradient(wynik.gradient * (x.dane > 0)) if wynik.gradient is not None and x.wymaga_gradientu else None
        return wynik


class LeakyReLU(Warstwa):
    """ReLU z małym nachyleniem dla wartości ujemnych."""

    def __init__(self, nachylenie=0.01):
        super().__init__()
        if nachylenie < 0:
            raise ValueError("nachylenie nie może być ujemne.")
        self.nachylenie = nachylenie

    def przepusc(self, x):
        dane = np.where(x.dane > 0, x.dane, self.nachylenie * x.dane)
        wynik = Tensor(dane, x.wymaga_gradientu, (x,))
        wynik._wstecz = lambda: x._dodaj_gradient(wynik.gradient * np.where(x.dane > 0, 1, self.nachylenie)) if wynik.gradient is not None and x.wymaga_gradientu else None
        return wynik


class Sigmoid(Warstwa):
    def __init__(self):
        super().__init__()

    def przepusc(self, x):
        dane = 1 / (1 + np.exp(-np.clip(x.dane, -20, 20)))
        wynik = Tensor(dane, x.wymaga_gradientu, (x,))
        wynik._wstecz = lambda: x._dodaj_gradient(wynik.gradient * dane * (1 - dane)) if wynik.gradient is not None and x.wymaga_gradientu else None
        return wynik


class Tanh(Warstwa):
    """Aktywacja hiperboliczna z zakresem od -1 do 1."""

    def __init__(self):
        super().__init__()

    def przepusc(self, x):
        dane = np.tanh(x.dane)
        wynik = Tensor(dane, x.wymaga_gradientu, (x,))
        wynik._wstecz = lambda: x._dodaj_gradient(wynik.gradient * (1 - dane ** 2)) if wynik.gradient is not None and x.wymaga_gradientu else None
        return wynik


class Softmax(Warstwa):
    def __init__(self):
        super().__init__()

    def przepusc(self, x):
        wykladniki = np.exp(x.dane - np.max(x.dane, axis=-1, keepdims=True))
        dane = wykladniki / wykladniki.sum(axis=-1, keepdims=True)
        wynik = Tensor(dane, x.wymaga_gradientu, (x,))

        def wstecz():
            if wynik.gradient is not None and x.wymaga_gradientu:
                x._dodaj_gradient(dane * (wynik.gradient - (wynik.gradient * dane).sum(axis=-1, keepdims=True)))

        wynik._wstecz = wstecz
        return wynik


class Dropout(Warstwa):
    """Losowo wyłącza część aktywacji wyłącznie podczas treningu."""

    def __init__(self, prawdopodobienstwo=0.5):
        super().__init__()
        if not 0 <= prawdopodobienstwo < 1:
            raise ValueError("prawdopodobienstwo musi należeć do zakresu [0, 1).")
        self.prawdopodobienstwo = prawdopodobienstwo
        self._generator = np.random.default_rng()

    def przepusc(self, x):
        if not self.szkolenie or self.prawdopodobienstwo == 0:
            return x
        maska = (self._generator.random(x.ksztalt) >= self.prawdopodobienstwo).astype(np.float32)
        maska /= 1 - self.prawdopodobienstwo
        wynik = Tensor(x.dane * maska, x.wymaga_gradientu, (x,))
        wynik._wstecz = lambda: x._dodaj_gradient(wynik.gradient * maska) if wynik.gradient is not None and x.wymaga_gradientu else None
        return wynik
