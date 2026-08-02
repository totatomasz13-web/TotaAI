from __future__ import annotations

import numpy as np


def _tensor(x) -> Tensor:
    return x if isinstance(x, Tensor) else Tensor(x)


class Tensor:
    """Tablica NumPy śledząca operacje i umożliwiająca automatyczne gradienty."""

    def __init__(self, dane, wymaga_gradientu: bool = False, _rodzice=(), _wstecz=None):
        self.dane = np.asarray(dane, dtype=np.float32)
        self.gradient: np.ndarray | None = None
        self.wymaga_gradientu = wymaga_gradientu
        self._rodzice = _rodzice
        self._wstecz = _wstecz or (lambda: None)

    @property
    def ksztalt(self) -> tuple[int, ...]:
        return self.dane.shape

    def wyzeruj_gradient(self):
        self.gradient = None

    def wstecz(self, gradient=None):
        topo, odwiedzone = [], set()

        def zbuduj(t):
            if id(t) not in odwiedzone:
                odwiedzone.add(id(t))
                for rodzic in t._rodzice:
                    zbuduj(rodzic)
                topo.append(t)

        zbuduj(self)
        self.gradient = np.ones_like(self.dane) if gradient is None else np.asarray(gradient, dtype=np.float32)
        for t in reversed(topo):
            t._wstecz()

    def __add__(self, other):
        other = _tensor(other)
        wynik = Tensor(self.dane + other.dane, self.wymaga_gradientu or other.wymaga_gradientu, (self, other))
        def wstecz():
            if wynik.gradient is not None:
                if self.wymaga_gradientu: self.gradient = _dodaj(self.gradient, _rozwin(wynik.gradient, self.dane.shape))
                if other.wymaga_gradientu: other.gradient = _dodaj(other.gradient, _rozwin(wynik.gradient, other.dane.shape))
        wynik._wstecz = wstecz
        return wynik

    __radd__ = __add__

    def __neg__(self):
        return self * -1

    def __sub__(self, other): return self + (-_tensor(other))
    def __rsub__(self, other): return _tensor(other) - self

    def __mul__(self, other):
        other = _tensor(other)
        wynik = Tensor(self.dane * other.dane, self.wymaga_gradientu or other.wymaga_gradientu, (self, other))
        def wstecz():
            if wynik.gradient is not None:
                if self.wymaga_gradientu: self.gradient = _dodaj(self.gradient, _rozwin(wynik.gradient * other.dane, self.dane.shape))
                if other.wymaga_gradientu: other.gradient = _dodaj(other.gradient, _rozwin(wynik.gradient * self.dane, other.dane.shape))
        wynik._wstecz = wstecz
        return wynik

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = _tensor(other)
        wynik = Tensor(self.dane / other.dane, self.wymaga_gradientu or other.wymaga_gradientu, (self, other))
        def wstecz():
            if wynik.gradient is not None:
                if self.wymaga_gradientu: self.gradient = _dodaj(self.gradient, _rozwin(wynik.gradient / other.dane, self.dane.shape))
                if other.wymaga_gradientu: other.gradient = _dodaj(other.gradient, _rozwin(-wynik.gradient * self.dane / (other.dane ** 2), other.dane.shape))
        wynik._wstecz = wstecz
        return wynik

    def __rtruediv__(self, other):
        return _tensor(other) / self

    def __matmul__(self, other):
        other = _tensor(other)
        wynik = Tensor(self.dane @ other.dane, self.wymaga_gradientu or other.wymaga_gradientu, (self, other))
        def wstecz():
            if wynik.gradient is not None:
                if self.wymaga_gradientu: self.gradient = _dodaj(self.gradient, wynik.gradient @ np.swapaxes(other.dane, -1, -2))
                if other.wymaga_gradientu: other.gradient = _dodaj(other.gradient, np.swapaxes(self.dane, -1, -2) @ wynik.gradient)
        wynik._wstecz = wstecz
        return wynik

    def suma(self):
        wynik = Tensor(self.dane.sum(), self.wymaga_gradientu, (self,))
        wynik._wstecz = lambda: self._dodaj_gradient(np.ones_like(self.dane) * wynik.gradient) if wynik.gradient is not None else None
        return wynik

    def srednia(self): return self.suma() * (1.0 / self.dane.size)

    def _dodaj_gradient(self, gradient): self.gradient = _dodaj(self.gradient, gradient)

    def __repr__(self): return f"Tensor(ksztalt={self.ksztalt}, dane={self.dane!r})"

    def __getstate__(self):
        return {"dane": self.dane, "gradient": self.gradient, "wymaga_gradientu": self.wymaga_gradientu}

    def __setstate__(self, state):
        self.dane = state["dane"]
        self.gradient = state["gradient"]
        self.wymaga_gradientu = state["wymaga_gradientu"]
        self._rodzice = ()
        self._wstecz = lambda: None


def _dodaj(stary, nowy): return nowy if stary is None else stary + nowy
def _rozwin(x, ksztalt):
    while x.ndim > len(ksztalt): x = x.sum(axis=0)
    for i, rozmiar in enumerate(ksztalt):
        if rozmiar == 1 and x.shape[i] != 1: x = x.sum(axis=i, keepdims=True)
    return x
