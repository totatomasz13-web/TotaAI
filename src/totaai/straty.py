import numpy as np
from .tensor import Tensor


class MSE:
    def __call__(self, przewidywane, oczekiwane):
        roznica = przewidywane - oczekiwane
        return (roznica * roznica).srednia()


class EntropiaKrzyzowa:
    def __call__(self, logits, klasy):
        klasy = klasy.dane.astype(int).reshape(-1)
        stabilne = logits.dane - np.max(logits.dane, axis=-1, keepdims=True)
        prawdopodobienstwa = np.exp(stabilne) / np.exp(stabilne).sum(axis=-1, keepdims=True)
        wartosc = -np.log(np.maximum(prawdopodobienstwa[np.arange(len(klasy)), klasy], 1e-12)).mean()
        wynik = Tensor(wartosc, logits.wymaga_gradientu, (logits,))
        def wstecz():
            if wynik.gradient is not None and logits.wymaga_gradientu:
                grad = prawdopodobienstwa.copy(); grad[np.arange(len(klasy)), klasy] -= 1; grad /= len(klasy)
                logits._dodaj_gradient(grad * wynik.gradient)
        wynik._wstecz = wstecz
        return wynik
