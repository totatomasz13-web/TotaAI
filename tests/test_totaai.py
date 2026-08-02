import numpy as np
import pytest

import totaai as ta


def test_autodiff_mnozenia():
    x = ta.Tensor([2.0], wymaga_gradientu=True)
    (x * x).wstecz()
    np.testing.assert_allclose(x.gradient, [4.0])


def test_model_uczy_xor():
    np.random.seed(4)
    model = ta.Model().dodaj(ta.WarstwaLiniowa(2, 8), ta.Sigmoid(), ta.WarstwaLiniowa(8, 1), ta.Sigmoid())
    model.skompiluj(ta.MSE(), ta.Adam(tempo=0.05))
    historia = model.trenuj([[0, 0], [0, 1], [1, 0], [1, 1]], [[0], [1], [1], [0]], epoki=40, pokazuj_postep=False)
    assert historia[-1] < historia[0]


def test_model_mozna_zapisac_i_wczytac(tmp_path):
    model = ta.Model().dodaj(ta.WarstwaLiniowa(2, 1))
    przed = model.przewidz([[1, 2]]).dane.copy()
    sciezka = tmp_path / "model.tota"
    model.zapisz(sciezka)
    po = ta.Model.wczytaj(sciezka).przewidz([[1, 2]]).dane
    np.testing.assert_allclose(przed, po)


def test_trening_partiami_i_podsumowanie():
    model = ta.Model().dodaj(ta.WarstwaLiniowa(2, 1))
    model.skompiluj(ta.MSE(), ta.SGD(tempo=0.05))
    historia = model.trenuj([[0, 0], [1, 1], [2, 2]], [[0], [2], [4]],
                            epoki=3, rozmiar_partii=2, tasuj=False, pokazuj_postep=False)
    assert len(historia) == 3
    assert "Razem:" in model.podsumowanie()
    assert model.ocen([[1, 1]], [[2]]) >= 0


def test_aktywacje_i_dropout():
    x = ta.Tensor([[-2.0, 0.0, 2.0]], wymaga_gradientu=True)
    np.testing.assert_allclose(ta.Tanh()(x).dane, np.tanh(x.dane))
    np.testing.assert_allclose(ta.LeakyReLU(0.1)(x).dane, [[-0.2, 0, 2]])
    dropout = ta.Dropout(0.5)
    dropout.ustaw_tryb(False)
    np.testing.assert_allclose(dropout(x).dane, x.dane)


def test_straty_metryki_i_podzial_danych():
    p = ta.Tensor([[0.9], [0.1]], wymaga_gradientu=True)
    y = ta.Tensor([[1], [0]])
    strata = ta.EntropiaBinarna()(p, y)
    strata.wstecz()
    assert strata.dane > 0
    assert p.gradient is not None
    assert ta.dokladnosc(p, y) == 1.0
    assert ta.blad_sredni_bezwzgledny(p, y) == pytest.approx(0.1)
    x_trening, x_test, y_trening, y_test = ta.podziel_dane([[1], [2], [3], [4]], [[0], [0], [1], [1]], ziarno=1)
    assert len(x_trening.dane) == len(y_trening.dane) == 3
    assert len(x_test.dane) == len(y_test.dane) == 1


def test_historia_walidacji_i_dzielenie_tensorow():
    model = ta.Model().dodaj(ta.WarstwaLiniowa(1, 1), ta.Dropout(0.2))
    model.skompiluj(ta.MSE(), ta.Adam(tempo=0.01))
    model.trenuj([[0], [1]], [[0], [1]], epoki=2, walidacja=([[0]], [[0]]), pokazuj_postep=False)
    assert len(model.historia["wal_strata"]) == 2
    x = ta.Tensor([4.0], wymaga_gradientu=True)
    (x / 2).wstecz()
    np.testing.assert_allclose(x.gradient, [0.5])
