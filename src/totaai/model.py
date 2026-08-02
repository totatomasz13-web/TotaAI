from __future__ import annotations

import pickle

import numpy as np

from .tensor import Tensor


class Model:
    def __init__(self):
        self.warstwy, self.strata, self.optymalizator = [], None, None
        self.szkolenie = True
        self.historia = {"strata": [], "wal_strata": []}
    def dodaj(self, *warstwy): self.warstwy.extend(warstwy); return self
    def skompiluj(self, strata, optymalizator): self.strata, self.optymalizator = strata, optymalizator; return self
    def ustaw_tryb(self, szkolenie=True):
        """Przełącza warstwy zależne od trybu, np. Dropout."""
        self.szkolenie = szkolenie
        for warstwa in self.warstwy:
            warstwa.ustaw_tryb(szkolenie)
        return self

    def przepusc(self, x):
        for warstwa in self.warstwy: x = warstwa(x)
        return x

    def przewidz(self, x):
        """Wykonuje predykcję w trybie ewaluacji, bez losowości Dropout."""
        poprzedni_tryb = self.szkolenie
        self.ustaw_tryb(False)
        wynik = self.przepusc(x if isinstance(x, Tensor) else Tensor(x))
        self.ustaw_tryb(poprzedni_tryb)
        return wynik
    def parametry(self):
        return [parametr for warstwa in self.warstwy for parametr in warstwa.parametry()]
    def trenuj(self, dane, etykiety, epoki=1, rozmiar_partii=None, tasuj=True, walidacja=None, pokazuj_postep=True):
        if self.strata is None or self.optymalizator is None: raise RuntimeError("Najpierw wywołaj skompiluj().")
        x_dane = (dane.dane if isinstance(dane, Tensor) else np.asarray(dane, dtype=np.float32))
        y_dane = (etykiety.dane if isinstance(etykiety, Tensor) else np.asarray(etykiety, dtype=np.float32))
        if len(x_dane) != len(y_dane): raise ValueError("dane i etykiety muszą mieć tyle samo przykładów.")
        if len(x_dane) == 0: raise ValueError("Zbiór treningowy nie może być pusty.")
        rozmiar_partii = len(x_dane) if rozmiar_partii is None else int(rozmiar_partii)
        if rozmiar_partii < 1: raise ValueError("rozmiar_partii musi być większy od zera.")
        generator = np.random.default_rng()
        self.ustaw_tryb(True)
        historia, wal_straty = [], []
        for epoka in range(epoki):
            indeksy = generator.permutation(len(x_dane)) if tasuj else np.arange(len(x_dane))
            straty_epoki = []
            for poczatek in range(0, len(x_dane), rozmiar_partii):
                partia = indeksy[poczatek:poczatek + rozmiar_partii]
                x, y = Tensor(x_dane[partia]), Tensor(y_dane[partia])
                self.optymalizator.wyzeruj_gradient(self.parametry())
                wynik = self.strata(self.przepusc(x), y); wynik.wstecz(); self.optymalizator.krok(self.parametry())
                straty_epoki.append(float(wynik.dane))
            historia.append(float(np.mean(straty_epoki)))
            if walidacja is not None:
                if len(walidacja) != 2:
                    raise ValueError("walidacja musi być parą: (dane, etykiety).")
                wal_straty.append(self.ocen(*walidacja))
            if pokazuj_postep and (epoka == 0 or (epoka + 1) % max(1, epoki // 10) == 0):
                tekst = f"Epoka {epoka + 1}/{epoki}: strata={historia[-1]:.6f}"
                if wal_straty:
                    tekst += f", wal_strata={wal_straty[-1]:.6f}"
                print(tekst)
        self.historia = {"strata": historia, "wal_strata": wal_straty}
        return historia

    def ocen(self, dane, etykiety):
        """Zwraca średnią wartość funkcji straty bez modyfikowania wag."""
        if self.strata is None: raise RuntimeError("Najpierw wywołaj skompiluj().")
        wynik = self.strata(self.przewidz(dane), etykiety if isinstance(etykiety, Tensor) else Tensor(etykiety))
        return float(wynik.dane)

    def podsumowanie(self):
        """Zwraca czytelny opis architektury i liczby parametrów."""
        linie, liczba = [], 0
        for numer, warstwa in enumerate(self.warstwy, 1):
            parametry = warstwa.parametry()
            warstwa_parametrow = sum(p.dane.size for p in parametry)
            liczba += warstwa_parametrow
            linie.append(f"{numer}. {warstwa.__class__.__name__} ({warstwa_parametrow} parametrów)")
        tekst = "\n".join(linie) + f"\nRazem: {liczba} parametrów"
        print(tekst)
        return tekst
    def zapisz(self, sciezka):
        with open(sciezka, "wb") as plik: pickle.dump(self, plik)
    @staticmethod
    def wczytaj(sciezka):
        with open(sciezka, "rb") as plik: return pickle.load(plik)
