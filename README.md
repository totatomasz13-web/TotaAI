# TotaAI

![Wersja](https://img.shields.io/pypi/v/totaai?label=PyPI)
![Testy](https://github.com/totatomasz13-web/TotaAI/actions/workflows/test.yml/badge.svg)
![Licencja MIT](https://img.shields.io/badge/licencja-MIT-b9ee5c)

**TotaAI** to czytelna biblioteka uczenia maszynowego z polskim API. Pozwala
budować i trenować proste sieci neuronowe bez zapamiętywania akademickiej
terminologii.

- Strona i tutoriale: https://totaai.pages.dev/
- Repozytorium: https://github.com/totatomasz13-web/TotaAI
- Dokumentacja API: [docs/API.md](docs/API.md)
- Instalacja: `pip install totaai`

```bash
pip install totaai
```

```python
import totaai as ta

model = ta.Model()
model.dodaj(ta.WarstwaLiniowa(2, 8), ta.ReLU(), ta.WarstwaLiniowa(8, 1))
model.skompiluj(ta.MSE(), ta.Adam(tempo=0.03))
model.trenuj(ta.Tensor([[0, 0], [0, 1], [1, 0], [1, 1]]),
             ta.Tensor([[0], [1], [1], [0]]), epoki=500,
             rozmiar_partii=4, pokazuj_postep=False)
print(model.przewidz(ta.Tensor([[1, 0]])).dane)
```

## Co działa w wersji 0.3

Rdzeń 0.2 zawiera tensor z automatycznym różniczkowaniem, warstwy liniowe, ReLU,
sigmoid, softmax, funkcje straty MSE i entropię krzyżową oraz SGD i Adam.
- Tensor NumPy z automatycznym różniczkowaniem.
- Warstwy: `WarstwaLiniowa`, `ReLU`, `Sigmoid`, `Softmax`, `Tanh`, `LeakyReLU`, `Dropout`.
- Funkcje straty: `MSE`, `MAE`, `EntropiaBinarna`, `EntropiaKrzyzowa`.
- Optymalizatory: `SGD`, `Adam`.
- Trening pełnym zbiorem lub partiami, tasowanie, walidacja i historia strat.
- Podsumowanie architektury oraz zapis/odczyt wag.
- Narzędzia: `podziel_dane`, `dokladnosc`, `blad_sredni_bezwzgledny`.
- CPU przez NumPy oraz opcjonalne CUDA przez CuPy.

## CUDA / GPU

CPU działa bez dodatkowych zależności. Dla GPU z CUDA 13:

```bash
pip install "totaai[cuda]"
```

```python
import totaai as ta

if ta.cuda_dostepna():
    model.na("cuda")
    wynik = model.przewidz([[1, 2]])
    print(wynik.urzadzenie)  # cuda
```

`Tensor(..., urzadzenie="cuda")` tworzy tensor GPU, a `tensor.numpy()`
kopiuje wynik do NumPy na CPU. CPU pozostaje domyślnym i w pełni wspieranym
trybem.

CNN, RNN, Transformery, GPU, autoenkodery i gotowe loadery danych nie są
jeszcze dostępne. Ich status będzie widoczny w dokumentacji i changelogu.

`Model.trenuj()` obsługuje batchowanie i tasowanie danych, a `ocen()` pozwala
sprawdzić stratę na osobnym zbiorze. `podsumowanie()` wypisuje architekturę i
liczbę parametrów modelu.

## Rozwój lokalny

```bash
pip install -e ".[dev]"
pytest
```

Strona działa w `website/`:

```bash
npm install
npm run dev
```

Pełna dokumentacja API znajduje się w [docs/API.md](docs/API.md) oraz na
stronie projektu: https://totaai.pages.dev/#dokumentacja.

## Struktura projektu

```text
src/totaai/          # biblioteka Python
tests/               # testy automatyczne
docs/API.md          # pełna dokumentacja API i tutoriale
website/             # strona TypeScript/Vite
.github/workflows/   # testy, wydania PyPI i deploy Cloudflare
```

## Wersjonowanie i wydania

Wersje są tworzone przez Release Please. Merge PR-a wydania aktualizuje
`pyproject.toml`, changelog i tag Git, a workflow publikacyjny wysyła paczkę
do PyPI. Aktualizacje zależności obsługuje Dependabot.

Zobacz [CONTRIBUTING.md](CONTRIBUTING.md), jeśli chcesz dodać funkcję lub
poprawić dokumentację.

## Licencja

MIT
