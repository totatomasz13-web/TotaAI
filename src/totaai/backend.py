"""Opcjonalny backend CUDA oparty na CuPy."""

import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None


def cuda_dostepna():
    return cp is not None and cp.cuda.runtime.getDeviceCount() > 0


def modul(urzadzenie):
    if urzadzenie == "cpu":
        return np
    if urzadzenie == "cuda":
        if not cuda_dostepna():
            raise RuntimeError('CUDA wymaga CuPy i dostępnego GPU. Zainstaluj: pip install "totaai[cuda]"')
        return cp
    raise ValueError('urzadzenie musi mieć wartość "cpu" albo "cuda".')


def urzadzenie_danych(dane):
    return "cuda" if cp is not None and isinstance(dane, cp.ndarray) else "cpu"


def do_numpy(dane):
    return cp.asnumpy(dane) if cp is not None and isinstance(dane, cp.ndarray) else np.asarray(dane)
