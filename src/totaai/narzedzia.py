"""Małe narzędzia do przygotowania danych i oceny predykcji."""

import numpy as np

from .tensor import Tensor


def podziel_dane(dane, etykiety, udzial_testowy=0.2, tasuj=True, ziarno=None):
    """Dzieli dane na trening, test oraz odpowiadające im etykiety."""
    x = dane.dane if isinstance(dane, Tensor) else np.asarray(dane, dtype=np.float32)
    y = etykiety.dane if isinstance(etykiety, Tensor) else np.asarray(etykiety, dtype=np.float32)
    if len(x) != len(y):
        raise ValueError("dane i etykiety muszą mieć tyle samo przykładów.")
    if not 0 < udzial_testowy < 1:
        raise ValueError("udzial_testowy musi należeć do zakresu (0, 1).")
    indeksy = np.arange(len(x))
    if tasuj:
        np.random.default_rng(ziarno).shuffle(indeksy)
    granica = int(len(x) * (1 - udzial_testowy))
    trening, test = indeksy[:granica], indeksy[granica:]
    return Tensor(x[trening]), Tensor(x[test]), Tensor(y[trening]), Tensor(y[test])


def dokladnosc(przewidywane, etykiety, prog=0.5):
    """Zwraca dokładność klasyfikacji binarnej albo wieloklasowej."""
    p = przewidywane.dane if isinstance(przewidywane, Tensor) else np.asarray(przewidywane)
    y = etykiety.dane if isinstance(etykiety, Tensor) else np.asarray(etykiety)
    if p.ndim > 1 and p.shape[-1] > 1:
        klasy = p.argmax(axis=-1)
    else:
        klasy = (p.reshape(-1) >= prog).astype(int)
    return float(np.mean(klasy == y.reshape(-1).astype(int)))


def blad_sredni_bezwzgledny(przewidywane, etykiety):
    """Zwraca MAE jako zwykłą liczbę, przydatną do raportowania."""
    p = przewidywane.dane if isinstance(przewidywane, Tensor) else np.asarray(przewidywane)
    y = etykiety.dane if isinstance(etykiety, Tensor) else np.asarray(etykiety)
    return float(np.abs(p - y).mean())
