import numpy as np

import totaai as ta


def test_tokenizer_znakowy_i_tokeny_specjalne():
    tokenizer = ta.Tokenizer("ab")
    zakodowane = tokenizer.encode("ab")
    assert zakodowane[0] == tokenizer.bos_id
    assert zakodowane[-1] == tokenizer.eos_id
    assert tokenizer.decode(zakodowane) == "ab"
    assert tokenizer.encode("?")[1] == tokenizer.unk_id


def test_transformer_ma_poprawny_ksztalt_i_gradienty():
    tokenizer = ta.Tokenizer("abc")
    model = ta.TransformerLM(tokenizer, dlugosc_sekwencji=4, rozmiar_embeddingu=8,
                             liczba_glow=2, liczba_blokow=1, rozmiar_feed_forward=16)
    x = np.array([tokenizer.encode("ab", dodaj_bos=False, dodaj_eos=False)], dtype=np.int64)
    logits = model(x)
    assert logits.ksztalt == (1, 2, tokenizer.vocab_size)
    ta.CrossEntropyLoss()(logits, ta.Tensor(x)).wstecz()
    assert all(parametr.gradient is not None for parametr in model.parametry())


def test_transformer_trenuje_krotki_tekst_i_generuje():
    tokenizer = ta.Tokenizer("abc ")
    model = ta.TransformerLM(tokenizer, dlugosc_sekwencji=4, rozmiar_embeddingu=8,
                             liczba_glow=2, liczba_blokow=1, rozmiar_feed_forward=16)
    historia = model.trenuj("abc abc abc", epoki=1, rozmiar_partii=2, pokazuj_postep=False)
    assert len(historia) == 1
    assert isinstance(model.generuj("a", maks_tokenow=2), str)
