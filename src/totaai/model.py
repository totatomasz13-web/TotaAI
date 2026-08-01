from __future__ import annotations
import pickle
from .tensor import Tensor


class Model:
    def __init__(self): self.warstwy, self.strata, self.optymalizator = [], None, None
    def dodaj(self, *warstwy): self.warstwy.extend(warstwy); return self
    def skompiluj(self, strata, optymalizator): self.strata, self.optymalizator = strata, optymalizator; return self
    def przepusc(self, x):
        for warstwa in self.warstwy: x = warstwa(x)
        return x
    def przewidz(self, x): return self.przepusc(x if isinstance(x, Tensor) else Tensor(x))
    def parametry(self):
        return [parametr for warstwa in self.warstwy for parametr in warstwa.parametry()]
    def trenuj(self, dane, etykiety, epoki=1, pokazuj_postep=True):
        if self.strata is None or self.optymalizator is None: raise RuntimeError("Najpierw wywołaj skompiluj().")
        x, y = dane if isinstance(dane, Tensor) else Tensor(dane), etykiety if isinstance(etykiety, Tensor) else Tensor(etykiety)
        historia = []
        for epoka in range(epoki):
            self.optymalizator.wyzeruj_gradient(self.parametry())
            wynik = self.strata(self.przepusc(x), y); wynik.wstecz(); self.optymalizator.krok(self.parametry())
            historia.append(float(wynik.dane))
            if pokazuj_postep and (epoka == 0 or (epoka + 1) % max(1, epoki // 10) == 0): print(f"Epoka {epoka + 1}/{epoki}: strata={historia[-1]:.6f}")
        return historia
    def zapisz(self, sciezka):
        with open(sciezka, "wb") as plik: pickle.dump(self, plik)
    @staticmethod
    def wczytaj(sciezka):
        with open(sciezka, "rb") as plik: return pickle.load(plik)
