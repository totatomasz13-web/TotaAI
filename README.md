# TotaAI

Prosta biblioteka uczenia maszynowego z polskim API.

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

## Status

Rdzeń 0.2 zawiera tensor z automatycznym różniczkowaniem, warstwy liniowe, ReLU,
sigmoid, softmax, funkcje straty MSE i entropię krzyżową oraz SGD i Adam.
Projekt jest rozwijany jako mała, czytelna biblioteka edukacyjna. Zaawansowane
CNN, RNN i Transformery będą dodawane jako kolejne moduły.

`Model.trenuj()` obsługuje batchowanie i tasowanie danych, a `ocen()` pozwala
sprawdzić stratę na osobnym zbiorze. `podsumowanie()` wypisuje architekturę i
liczbę parametrów modelu.

## Rozwój

```bash
pip install -e ".[dev]"
pytest
```

Strona działa w `website/`:

```bash
npm install
npm run dev
```

## Licencja

MIT
