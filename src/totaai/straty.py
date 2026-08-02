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
        wynik = Tensor(np.abs(roznica).mean(), przewidywane.wymaga_gradientu, (przewidywane,))
        wynik._wstecz = lambda: przewidywane._dodaj_gradient(np.sign(roznica) * wynik.gradient / roznica.size) if wynik.gradient is not None and przewidywane.wymaga_gradientu else None
        return wynik


class EntropiaBinarna:
    """Binary cross-entropy dla prawdopodobieństw z Sigmoid."""

    def __call__(self, prawdopodobienstwa, etykiety):
        y = etykiety.dane
        p = np.clip(prawdopodobienstwa.dane, 1e-7, 1 - 1e-7)
        wartosc = -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()
        wynik = Tensor(wartosc, prawdopodobienstwa.wymaga_gradientu, (prawdopodobienstwa,))
        wynik._wstecz = lambda: prawdopodobienstwa._dodaj_gradient(((p - y) / (p * (1 - p))) * wynik.gradient / y.size) if wynik.gradient is not None and prawdopodobienstwa.wymaga_gradientu else None
        return wynik


class EntropiaKrzyzowa:
    """Stabilna numerycznie entropia krzyżowa dla logitów i indeksów klas."""

    def __call__(self, logits, klasy):
        klasy = klasy.dane.astype(int).reshape(-1)
        stabilne = logits.dane - np.max(logits.dane, axis=-1, keepdims=True)
        prawdopodobienstwa = np.exp(stabilne) / np.exp(stabilne).sum(axis=-1, keepdims=True)
        wartosc = -np.log(np.maximum(prawdopodobienstwa[np.arange(len(klasy)), klasy], 1e-12)).mean()
        wynik = Tensor(wartosc, logits.wymaga_gradientu, (logits,))

        def wstecz():
            if wynik.gradient is not None and logits.wymaga_gradientu:
                grad = prawdopodobienstwa.copy()
                grad[np.arange(len(klasy)), klasy] -= 1
                logits._dodaj_gradient(grad * wynik.gradient / len(klasy))

        wynik._wstecz = wstecz
        return wynik
