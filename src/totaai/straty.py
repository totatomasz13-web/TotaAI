import numpy as np

from .tensor import Tensor


class MSE:
    """Średni błąd kwadratowy, przydatny w regresji."""

    def __call__(self, przewidywane, oczekiwane):
        roznica = przewidywane - oczekiwane
        return (roznica * roznica).srednia()


class MAE:
    """Średni błąd bezwzględny, mniej wrażliwy na wartości odstające niż MSE."""

    def __call__(self, przewidywane, oczekiwane):
        roznica = przewidywane.dane - oczekiwane.dane
        wynik = Tensor(przewidywane.modul.abs(roznica).mean(), przewidywane.wymaga_gradientu, (przewidywane,), urzadzenie=przewidywane.urzadzenie)
        wynik._wstecz = lambda: przewidywane._dodaj_gradient(np.sign(roznica) * wynik.gradient / roznica.size) if wynik.gradient is not None and przewidywane.wymaga_gradientu else None
        return wynik


class EntropiaBinarna:
    """Binary cross-entropy dla prawdopodobieństw z Sigmoid."""

    def __call__(self, prawdopodobienstwa, etykiety):
        y = etykiety.dane
        xp = prawdopodobienstwa.modul; p = xp.clip(prawdopodobienstwa.dane, 1e-7, 1 - 1e-7)
        wartosc = -(y * xp.log(p) + (1 - y) * xp.log(1 - p)).mean()
        wynik = Tensor(wartosc, prawdopodobienstwa.wymaga_gradientu, (prawdopodobienstwa,), urzadzenie=prawdopodobienstwa.urzadzenie)
        wynik._wstecz = lambda: prawdopodobienstwa._dodaj_gradient(((p - y) / (p * (1 - p))) * wynik.gradient / y.size) if wynik.gradient is not None and prawdopodobienstwa.wymaga_gradientu else None
        return wynik


class EntropiaKrzyzowa:
    """Stabilna numerycznie entropia krzyżowa dla logitów i indeksów klas."""

    def __call__(self, logits, klasy):
        klasy = klasy.dane.astype(int).reshape(-1)
        xp = logits.modul; klasy = xp.asarray(klasy)
        stabilne = logits.dane - xp.max(logits.dane, axis=-1, keepdims=True)
        prawdopodobienstwa = xp.exp(stabilne) / xp.exp(stabilne).sum(axis=-1, keepdims=True)
        wartosc = -xp.log(xp.maximum(prawdopodobienstwa[xp.arange(len(klasy)), klasy], 1e-12)).mean()
        wynik = Tensor(wartosc, logits.wymaga_gradientu, (logits,), urzadzenie=logits.urzadzenie)

        def wstecz():
            if wynik.gradient is not None and logits.wymaga_gradientu:
                grad = prawdopodobienstwa.copy()
                grad[xp.arange(len(klasy)), klasy] -= 1
                logits._dodaj_gradient(grad * wynik.gradient / len(klasy))

        wynik._wstecz = wstecz
        return wynik
