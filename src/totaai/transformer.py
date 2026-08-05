"""Mały, edukacyjny Transformer do pracy z tekstem.

Moduł korzysta z tego samego autodiff co klasyczne warstwy TotaAI. Nie jest
zamiennikiem dużych frameworków, ale pozwala zbudować i trenować mały model
językowy bez zmiany dotychczasowego API biblioteki.
"""

from __future__ import annotations

import pickle
from math import sqrt

import numpy as np

from .backend import modul
from .optymalizatory import Adam
from .straty import EntropiaKrzyzowa
from .tensor import Tensor


def _unary(x, dane, pochodna):
    wynik = Tensor(dane, x.wymaga_gradientu, (x,), urzadzenie=x.urzadzenie)
    wynik._wstecz = lambda: x._dodaj_gradient(wynik.gradient * pochodna) if wynik.gradient is not None and x.wymaga_gradientu else None
    return wynik


def _softmax(x, os=-1):
    xp = x.modul
    przesuniete = x.dane - xp.max(x.dane, axis=os, keepdims=True)
    dane = xp.exp(przesuniete)
    dane /= dane.sum(axis=os, keepdims=True)
    wynik = Tensor(dane, x.wymaga_gradientu, (x,), urzadzenie=x.urzadzenie)

    def wstecz():
        if wynik.gradient is not None and x.wymaga_gradientu:
            suma = (wynik.gradient * dane).sum(axis=os, keepdims=True)
            x._dodaj_gradient(dane * (wynik.gradient - suma))

    wynik._wstecz = wstecz
    return wynik


class Tokenizer:
    """Deterministyczny tokenizer znakowy z tokenami specjalnymi."""

    SPECJALNE = ("<PAD>", "<UNK>", "<BOS>", "<EOS>")

    def __init__(self, tekst=None, znaki=None):
        znaki = list(znaki if znaki is not None else sorted(set(tekst or "")))
        self.token_to_id = {token: i for i, token in enumerate(self.SPECJALNE)}
        self.token_to_id.update({znak: i + len(self.SPECJALNE) for i, znak in enumerate(znaki) if znak not in self.token_to_id})
        self.id_to_token = {i: token for token, i in self.token_to_id.items()}

    @property
    def vocab_size(self): return len(self.token_to_id)
    @property
    def slownik(self): return dict(self.token_to_id)
    @property
    def pad_id(self): return self.token_to_id["<PAD>"]
    @property
    def unk_id(self): return self.token_to_id["<UNK>"]
    @property
    def bos_id(self): return self.token_to_id["<BOS>"]
    @property
    def eos_id(self): return self.token_to_id["<EOS>"]

    def encode(self, tekst, dodaj_bos=True, dodaj_eos=True):
        tokeny = [self.token_to_id.get(znak, self.unk_id) for znak in tekst]
        if dodaj_bos: tokeny.insert(0, self.bos_id)
        if dodaj_eos: tokeny.append(self.eos_id)
        return tokeny

    def decode(self, tokeny, pomijaj_specjalne=True):
        wynik = []
        for token in tokeny:
            znak = self.id_to_token.get(int(token), "<UNK>")
            if pomijaj_specjalne and znak in self.SPECJALNE: continue
            wynik.append(znak)
        return "".join(wynik)

    zakoduj = encode
    odkoduj = decode

    def zapisz(self, sciezka):
        with open(sciezka, "wb") as plik: pickle.dump(self, plik)

    @staticmethod
    def wczytaj(sciezka):
        with open(sciezka, "rb") as plik: return pickle.load(plik)


class Embedding:
    def __init__(self, liczba_tokenow, rozmiar, urzadzenie="cpu", ziarno=None):
        xp = modul(urzadzenie)
        if ziarno is not None: xp.random.seed(ziarno)
        self.wagi = Tensor(xp.random.normal(0, 1 / sqrt(rozmiar), (liczba_tokenow, rozmiar)), True, urzadzenie=urzadzenie)

    def __call__(self, tokeny):
        indeksy = np.asarray(tokeny.dane if isinstance(tokeny, Tensor) else tokeny, dtype=np.int64)
        if indeksy.ndim == 1: indeksy = indeksy[None, :]
        dane = self.wagi.dane[indeksy]
        wynik = Tensor(dane, True, (self.wagi,), urzadzenie=self.wagi.urzadzenie)

        def wstecz():
            if wynik.gradient is None: return
            grad = self.wagi.modul.zeros_like(self.wagi.dane)
            self.wagi.modul.add.at(grad, indeksy, wynik.gradient)
            self.wagi._dodaj_gradient(grad)

        wynik._wstecz = wstecz
        return wynik

    def parametry(self): return [self.wagi]


class PositionalEncoding:
    def __init__(self, maks_dlugosc, rozmiar, urzadzenie="cpu", uczone=True):
        xp = modul(urzadzenie)
        pozycje = xp.arange(maks_dlugosc)[:, None]
        dzielnik = xp.exp(xp.arange(0, rozmiar, 2) * (-np.log(10000.0) / rozmiar))
        kod = xp.zeros((maks_dlugosc, rozmiar), dtype=xp.float32)
        kod[:, 0::2] = xp.sin(pozycje * dzielnik)
        kod[:, 1::2] = xp.cos(pozycje * dzielnik[:kod[:, 1::2].shape[1]])
        self.kod = Tensor(kod, uczone, urzadzenie=urzadzenie)
        self.maks_dlugosc = maks_dlugosc

    def __call__(self, x):
        dlugosc = x.ksztalt[-2]
        if dlugosc > self.maks_dlugosc: raise ValueError("Sekwencja jest dłuższa niż maks_dlugosc.")
        return x + self.kod.reshape((1, self.maks_dlugosc, self.kod.ksztalt[-1]))[:, :dlugosc]

    def parametry(self): return [self.kod] if self.kod.wymaga_gradientu else []


class Linear:
    def __init__(self, wejscia, wyjscia, urzadzenie="cpu"):
        xp = modul(urzadzenie)
        self.wagi = Tensor(xp.random.normal(0, sqrt(2 / wejscia), (wejscia, wyjscia)), True, urzadzenie=urzadzenie)
        self.bias = Tensor(xp.zeros(wyjscia), True, urzadzenie=urzadzenie)

    def __call__(self, x):
        dane = x.dane @ self.wagi.dane + self.bias.dane
        wynik = Tensor(dane, True, (x, self.wagi, self.bias), urzadzenie=x.urzadzenie)

        def wstecz():
            if wynik.gradient is None: return
            if x.wymaga_gradientu: x._dodaj_gradient(wynik.gradient @ self.wagi.dane.T)
            if self.wagi.wymaga_gradientu: self.wagi._dodaj_gradient(x.dane.reshape((-1, x.ksztalt[-1])).T @ wynik.gradient.reshape((-1, wynik.ksztalt[-1])))
            if self.bias.wymaga_gradientu: self.bias._dodaj_gradient(wynik.gradient.reshape((-1, wynik.ksztalt[-1])).sum(axis=0))

        wynik._wstecz = wstecz
        return wynik

    def parametry(self): return [self.wagi, self.bias]


class LayerNorm:
    def __init__(self, rozmiar, epsilon=1e-5, urzadzenie="cpu"):
        xp = modul(urzadzenie)
        self.gamma = Tensor(xp.ones(rozmiar), True, urzadzenie=urzadzenie)
        self.beta = Tensor(xp.zeros(rozmiar), True, urzadzenie=urzadzenie)
        self.epsilon = epsilon

    def __call__(self, x):
        xp = x.modul; srednia = x.dane.mean(axis=-1, keepdims=True)
        wariancja = ((x.dane - srednia) ** 2).mean(axis=-1, keepdims=True)
        odwrotnosc = 1 / xp.sqrt(wariancja + self.epsilon)
        znormalizowane = (x.dane - srednia) * odwrotnosc
        dane = znormalizowane * self.gamma.dane + self.beta.dane
        wynik = Tensor(dane, True, (x, self.gamma, self.beta), urzadzenie=x.urzadzenie)

        def wstecz():
            if wynik.gradient is None: return
            n = x.ksztalt[-1]
            if x.wymaga_gradientu:
                g = wynik.gradient * self.gamma.dane
                x._dodaj_gradient((g - g.mean(axis=-1, keepdims=True) - znormalizowane * (g * znormalizowane).mean(axis=-1, keepdims=True)) * odwrotnosc)
            if self.gamma.wymaga_gradientu: self.gamma._dodaj_gradient((wynik.gradient * znormalizowane).sum(axis=tuple(range(wynik.gradient.ndim - 1))))
            if self.beta.wymaga_gradientu: self.beta._dodaj_gradient(wynik.gradient.sum(axis=tuple(range(wynik.gradient.ndim - 1))))

        wynik._wstecz = wstecz
        return wynik

    def parametry(self): return [self.gamma, self.beta]


class SelfAttention:
    def __init__(self, rozmiar, urzadzenie="cpu"):
        self.query, self.key, self.value = (Linear(rozmiar, rozmiar, urzadzenie) for _ in range(3))
        self.rozmiar = rozmiar

    def __call__(self, x):
        q, k, v = self.query(x), self.key(x), self.value(x)
        skale = q @ k.transpose(0, 2, 1) * (1 / sqrt(self.rozmiar))
        maska = np.triu(np.full(skale.ksztalt[-2:], -1e9, dtype=np.float32), 1)
        return _softmax(skale + maska) @ v

    def parametry(self): return self.query.parametry() + self.key.parametry() + self.value.parametry()


class MultiHeadAttention:
    def __init__(self, rozmiar, liczba_glow, urzadzenie="cpu"):
        if rozmiar % liczba_glow: raise ValueError("rozmiar musi dzielić się przez liczba_glow.")
        self.projekcja_qkv = Linear(rozmiar, rozmiar * 3, urzadzenie)
        self.projekcja = Linear(rozmiar, rozmiar, urzadzenie)
        self.rozmiar, self.liczba_glow, self.rozmiar_glowy = rozmiar, liczba_glow, rozmiar // liczba_glow

    def __call__(self, x):
        polaczone = self.projekcja_qkv(x)
        rozmiar = self.rozmiar
        q, k, v = polaczone[..., :rozmiar], polaczone[..., rozmiar:2 * rozmiar], polaczone[..., 2 * rozmiar:]
        ksztalt = (x.ksztalt[0], self.liczba_glow, x.ksztalt[1], self.rozmiar_glowy)
        q, k, v = (z.reshape((x.ksztalt[0], x.ksztalt[1], self.liczba_glow, self.rozmiar_glowy)).transpose(0, 2, 1, 3) for z in (q, k, v))
        wyniki = q @ k.transpose(0, 1, 3, 2) * (1 / sqrt(self.rozmiar_glowy))
        maska = np.triu(np.full((x.ksztalt[1], x.ksztalt[1]), -1e9, dtype=np.float32), 1)
        wyniki = _softmax(wyniki + maska) @ v
        return self.projekcja(wyniki.transpose(0, 2, 1, 3).reshape((x.ksztalt[0], x.ksztalt[1], self.rozmiar)))

    def parametry(self): return self.projekcja_qkv.parametry() + self.projekcja.parametry()


class FeedForward:
    def __init__(self, rozmiar, rozmiar_ukryty, urzadzenie="cpu"):
        self.pierwsza, self.druga = Linear(rozmiar, rozmiar_ukryty, urzadzenie), Linear(rozmiar_ukryty, rozmiar, urzadzenie)

    def __call__(self, x):
        ukryte = self.pierwsza(x)
        xp = ukryte.modul; dane = 0.5 * ukryte.dane * (1 + xp.tanh(sqrt(2 / np.pi) * (ukryte.dane + 0.044715 * ukryte.dane ** 3)))
        aktywacja = _unary(ukryte, dane, 0.5 * (1 + xp.tanh(sqrt(2 / np.pi) * (ukryte.dane + 0.044715 * ukryte.dane ** 3))))
        return self.druga(aktywacja)

    def parametry(self): return self.pierwsza.parametry() + self.druga.parametry()


class TransformerBlock:
    def __init__(self, rozmiar, liczba_glow=4, rozmiar_feed_forward=None, urzadzenie="cpu"):
        self.norm1 = LayerNorm(rozmiar, urzadzenie=urzadzenie)
        self.attention = MultiHeadAttention(rozmiar, liczba_glow, urzadzenie)
        self.norm2 = LayerNorm(rozmiar, urzadzenie=urzadzenie)
        self.feed_forward = FeedForward(rozmiar, rozmiar_feed_forward or rozmiar * 4, urzadzenie)

    def __call__(self, x):
        po_attention = x + self.attention(self.norm1(x))
        return po_attention + self.feed_forward(self.norm2(po_attention))

    def parametry(self): return self.norm1.parametry() + self.attention.parametry() + self.norm2.parametry() + self.feed_forward.parametry()


class CrossEntropyLoss(EntropiaKrzyzowa):
    """Cross-entropy dla logitów o kształcie (batch, sekwencja, słownik)."""

    def __init__(self, ignore_index=None): self.ignore_index = ignore_index

    def __call__(self, logits, cele):
        rozmiar = logits.ksztalt[-1]
        logits, cele = logits.reshape((-1, rozmiar)), cele.reshape((-1,))
        if self.ignore_index is not None:
            maska = cele.dane.astype(int) != self.ignore_index
            if not np.any(maska): raise ValueError("Wszystkie cele mają ignore_index.")
            logits, cele = logits[maska], cele[maska]
        return super().__call__(logits, cele)


class AdamW(Adam):
    def __init__(self, tempo=0.001, weight_decay=0.01, **kwargs):
        super().__init__(tempo=tempo, **kwargs); self.weight_decay = weight_decay

    def krok(self, parametry):
        super().krok(parametry)
        for parametr in parametry: parametr.dane *= 1 - self.tempo * self.weight_decay


class TransformerLM:
    def __init__(self, tokenizer, dlugosc_sekwencji=32, rozmiar_embeddingu=64, liczba_glow=4, liczba_blokow=2, rozmiar_feed_forward=128, urzadzenie="cpu"):
        self.tokenizer = tokenizer
        self.dlugosc_sekwencji, self.urzadzenie = dlugosc_sekwencji, urzadzenie
        self.embedding = Embedding(tokenizer.vocab_size, rozmiar_embeddingu, urzadzenie)
        self.pozycja = PositionalEncoding(dlugosc_sekwencji, rozmiar_embeddingu, urzadzenie, uczone=False)
        self.bloki = [TransformerBlock(rozmiar_embeddingu, liczba_glow, rozmiar_feed_forward, urzadzenie) for _ in range(liczba_blokow)]
        self.norm = LayerNorm(rozmiar_embeddingu, urzadzenie=urzadzenie)
        self.glowa = Linear(rozmiar_embeddingu, tokenizer.vocab_size, urzadzenie)
        self.szkolenie = True

    def __call__(self, tokeny):
        tokeny = np.asarray(tokeny, dtype=np.int64) if not isinstance(tokeny, Tensor) else tokeny
        x = self.pozycja(self.embedding(tokeny))
        for blok in self.bloki: x = blok(x)
        return self.glowa(self.norm(x))

    def parametry(self):
        wynik = self.embedding.parametry()
        for element in self.bloki: wynik.extend(element.parametry())
        wynik.extend(self.norm.parametry() + self.glowa.parametry())
        return wynik

    def generuj(self, tekst, tokenizer=None, maks_tokenow=50, temperatura=1.0, top_k=None, top_p=None):
        tokenizer = tokenizer or self.tokenizer
        tokeny = tokenizer.encode(tekst, dodaj_bos=False, dodaj_eos=False)
        for _ in range(maks_tokenow):
            kontekst = tokeny[-self.dlugosc_sekwencji:]
            prawdopodobienstwa = self(Tensor([kontekst])).dane[0, -1].copy()
            prawdopodobienstwa = np.exp((prawdopodobienstwa - prawdopodobienstwa.max()) / max(temperatura, 1e-6))
            prawdopodobienstwa /= prawdopodobienstwa.sum()
            if top_k:
                odciete = np.argpartition(prawdopodobienstwa, -top_k)[:-top_k]
                prawdopodobienstwa[odciete] = 0
            if top_p:
                kolejnosc = np.argsort(prawdopodobienstwa)[::-1]
                maska = np.cumsum(prawdopodobienstwa[kolejnosc]) > top_p
                maska[0] = False
                prawdopodobienstwa[kolejnosc[maska]] = 0
            prawdopodobienstwa /= prawdopodobienstwa.sum()
            nastepny = int(np.random.choice(len(prawdopodobienstwa), p=prawdopodobienstwa))
            tokeny.append(nastepny)
            if nastepny == tokenizer.eos_id: break
        return tokenizer.decode(tokeny)

    def trenuj(self, tekst, epoki=1, rozmiar_partii=8, tempo=0.001, warmup=0, checkpoint=None, pokazuj_postep=True, **_):
        tokeny = np.asarray(self.tokenizer.encode(tekst), dtype=np.int64)
        if len(tokeny) < 2: raise ValueError("Tekst treningowy musi mieć co najmniej dwa tokeny.")
        liczba_przykladow = max(1, len(tokeny) - self.dlugosc_sekwencji)
        wejscia = np.full((liczba_przykladow, self.dlugosc_sekwencji), self.tokenizer.pad_id, dtype=np.int64)
        cele = np.full_like(wejscia, self.tokenizer.pad_id)
        for i in range(liczba_przykladow):
            fragment_wejscia = tokeny[i:i + self.dlugosc_sekwencji]
            fragment_celu = tokeny[i + 1:i + self.dlugosc_sekwencji + 1]
            wejscia[i, :len(fragment_wejscia)] = fragment_wejscia
            cele[i, :len(fragment_celu)] = fragment_celu
        optymalizator = AdamW(tempo=tempo)
        historia = []
        for epoka in range(epoki):
            if warmup:
                optymalizator.tempo = tempo * min(1.0, (epoka + 1) / warmup)
            straty = []
            for poczatek in range(0, len(wejscia), rozmiar_partii):
                x = Tensor(wejscia[poczatek:poczatek + rozmiar_partii])
                y = Tensor(cele[poczatek:poczatek + rozmiar_partii])
                optymalizator.wyzeruj_gradient(self.parametry())
                strata = CrossEntropyLoss(ignore_index=self.tokenizer.pad_id)(self(x), y); strata.wstecz()
                for parametr in self.parametry():
                    if parametr.gradient is not None:
                        norma = np.sqrt(np.sum(parametr.gradient ** 2))
                        if norma > 1.0: parametr.gradient *= 1.0 / norma
                optymalizator.krok(self.parametry()); straty.append(float(strata.dane))
            historia.append(float(np.mean(straty)))
            if checkpoint is not None:
                self.zapisz(checkpoint)
            if pokazuj_postep: print(f"Epoka {epoka + 1}/{epoki}: strata={historia[-1]:.6f}")
        return historia

    def zapisz(self, sciezka):
        with open(sciezka, "wb") as plik: pickle.dump(self, plik)

    @staticmethod
    def wczytaj(sciezka):
        with open(sciezka, "rb") as plik: return pickle.load(plik)
