from __future__ import annotations

from .backend import do_numpy, modul, urzadzenie_danych


def _tensor(x, urzadzenie):
    return x if isinstance(x, Tensor) else Tensor(x, urzadzenie=urzadzenie)


class Tensor:
    """Tensor CPU (NumPy) lub CUDA (CuPy) z automatycznym różniczkowaniem."""

    def __init__(self, dane, wymaga_gradientu=False, _rodzice=(), _wstecz=None, urzadzenie=None):
        self.urzadzenie = urzadzenie or urzadzenie_danych(dane)
        self.modul = modul(self.urzadzenie)
        self.dane = self.modul.asarray(dane, dtype=self.modul.float32)
        self.gradient = None
        self.wymaga_gradientu = wymaga_gradientu
        self._rodzice, self._wstecz = _rodzice, _wstecz or (lambda: None)

    @property
    def ksztalt(self): return self.dane.shape
    def numpy(self): return do_numpy(self.dane)
    def na(self, urzadzenie): return Tensor(self.dane, self.wymaga_gradientu, urzadzenie=urzadzenie)
    def przenies(self, urzadzenie):
        nowy = self.na(urzadzenie); self.dane, self.gradient, self.urzadzenie, self.modul = nowy.dane, None, nowy.urzadzenie, nowy.modul; return self
    def wyzeruj_gradient(self): self.gradient = None

    def wstecz(self, gradient=None):
        topo, odwiedzone = [], set()
        def zbuduj(t):
            if id(t) not in odwiedzone:
                odwiedzone.add(id(t)); [zbuduj(r) for r in t._rodzice]; topo.append(t)
        zbuduj(self)
        self.gradient = self.modul.ones_like(self.dane) if gradient is None else self.modul.asarray(gradient, dtype=self.modul.float32)
        for t in reversed(topo): t._wstecz()

    def _binarny(self, other, operacja, grad_a, grad_b):
        other = _tensor(other, self.urzadzenie)
        if other.urzadzenie != self.urzadzenie: raise ValueError("Tensory muszą być na tym samym urządzeniu.")
        wynik = Tensor(operacja(self.dane, other.dane), self.wymaga_gradientu or other.wymaga_gradientu, (self, other), urzadzenie=self.urzadzenie)
        def wstecz():
            if wynik.gradient is not None:
                if self.wymaga_gradientu: self._dodaj_gradient(_rozwin(grad_a(wynik.gradient, self.dane, other.dane), self.dane.shape))
                if other.wymaga_gradientu: other._dodaj_gradient(_rozwin(grad_b(wynik.gradient, self.dane, other.dane), other.dane.shape))
        wynik._wstecz = wstecz; return wynik

    def __add__(self, other): return self._binarny(other, lambda a,b:a+b, lambda g,a,b:g, lambda g,a,b:g)
    __radd__ = __add__
    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-_tensor(other, self.urzadzenie))
    def __rsub__(self, other): return _tensor(other, self.urzadzenie) - self
    def __mul__(self, other): return self._binarny(other, lambda a,b:a*b, lambda g,a,b:g*b, lambda g,a,b:g*a)
    __rmul__ = __mul__
    def __truediv__(self, other): return self._binarny(other, lambda a,b:a/b, lambda g,a,b:g/b, lambda g,a,b:-g*a/(b**2))
    def __rtruediv__(self, other): return _tensor(other, self.urzadzenie) / self
    def __matmul__(self, other):
        other = _tensor(other, self.urzadzenie)
        if other.urzadzenie != self.urzadzenie: raise ValueError("Tensory muszą być na tym samym urządzeniu.")
        wynik = Tensor(self.dane @ other.dane, self.wymaga_gradientu or other.wymaga_gradientu, (self, other), urzadzenie=self.urzadzenie)
        def wstecz():
            if wynik.gradient is not None:
                if self.wymaga_gradientu: self._dodaj_gradient(wynik.gradient @ self.modul.swapaxes(other.dane, -1, -2))
                if other.wymaga_gradientu: other._dodaj_gradient(self.modul.swapaxes(self.dane, -1, -2) @ wynik.gradient)
        wynik._wstecz = wstecz; return wynik
    def __getitem__(self, indeks):
        dane = self.dane[indeks]
        wynik = Tensor(dane, self.wymaga_gradientu, (self,), urzadzenie=self.urzadzenie)
        def wstecz():
            if wynik.gradient is not None and self.wymaga_gradientu:
                gradient = self.modul.zeros_like(self.dane)
                gradient[indeks] += wynik.gradient
                self._dodaj_gradient(gradient)
        wynik._wstecz = wstecz
        return wynik
    def reshape(self, *ksztalt):
        if len(ksztalt) == 1 and isinstance(ksztalt[0], (tuple, list)):
            ksztalt = tuple(ksztalt[0])
        wynik = Tensor(self.dane.reshape(*ksztalt), self.wymaga_gradientu, (self,), urzadzenie=self.urzadzenie)
        wynik._wstecz = lambda: self._dodaj_gradient(wynik.gradient.reshape(self.ksztalt)) if wynik.gradient is not None and self.wymaga_gradientu else None
        return wynik
    def transpose(self, *osie):
        osie = osie or tuple(reversed(range(self.dane.ndim)))
        wynik = Tensor(self.modul.transpose(self.dane, osie), self.wymaga_gradientu, (self,), urzadzenie=self.urzadzenie)
        odwrotne = tuple(osie.index(i) for i in range(len(osie)))
        wynik._wstecz = lambda: self._dodaj_gradient(self.modul.transpose(wynik.gradient, odwrotne)) if wynik.gradient is not None and self.wymaga_gradientu else None
        return wynik
    @property
    def T(self): return self.transpose()
    def exp(self):
        dane = self.modul.exp(self.dane)
        wynik = Tensor(dane, self.wymaga_gradientu, (self,), urzadzenie=self.urzadzenie)
        wynik._wstecz = lambda: self._dodaj_gradient(wynik.gradient * dane) if wynik.gradient is not None and self.wymaga_gradientu else None
        return wynik
    def log(self):
        dane = self.modul.log(self.dane)
        wynik = Tensor(dane, self.wymaga_gradientu, (self,), urzadzenie=self.urzadzenie)
        wynik._wstecz = lambda: self._dodaj_gradient(wynik.gradient / self.dane) if wynik.gradient is not None and self.wymaga_gradientu else None
        return wynik
    def suma(self):
        wynik = Tensor(self.dane.sum(), self.wymaga_gradientu, (self,), urzadzenie=self.urzadzenie)
        wynik._wstecz = lambda: self._dodaj_gradient(self.modul.ones_like(self.dane) * wynik.gradient) if wynik.gradient is not None and self.wymaga_gradientu else None
        return wynik
    def srednia(self): return self.suma() * (1.0 / self.dane.size)
    def _dodaj_gradient(self, gradient): self.gradient = gradient if self.gradient is None else self.gradient + gradient
    def __repr__(self): return f"Tensor(urzadzenie={self.urzadzenie!r}, ksztalt={self.ksztalt})"
    def __getstate__(self): return {"dane": self.numpy(), "gradient": None if self.gradient is None else do_numpy(self.gradient), "wymaga_gradientu": self.wymaga_gradientu}
    def __setstate__(self, state): self.__init__(state["dane"], state["wymaga_gradientu"]); self.gradient = state["gradient"]


def _rozwin(x, ksztalt):
    while x.ndim > len(ksztalt): x = x.sum(axis=0)
    for i, rozmiar in enumerate(ksztalt):
        if rozmiar == 1 and x.shape[i] != 1: x = x.sum(axis=i, keepdims=True)
    return x
