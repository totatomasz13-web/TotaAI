import numpy as np
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
