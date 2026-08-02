# Dokumentacja TotaAI

TotaAI to edukacyjna biblioteka uczenia maszynowego z prostym, polskim API.
Aktualna wersja udostępnia rdzeń oparty na NumPy: tensory, automatyczne
różniczkowanie, warstwy gęste, aktywacje, funkcje straty, optymalizatory i
trening modeli.

## Instalacja

```bash
pip install totaai
```

Wersja deweloperska:

```bash
git clone https://github.com/totatomasz13-web/TotaAI.git
cd TotaAI
pip install -e ".[dev]"
pytest
```

## Pierwszy model

```python
import totaai as ta

model = ta.Model()
model.dodaj(
    ta.WarstwaLiniowa(2, 8),
    ta.Sigmoid(),
    ta.WarstwaLiniowa(8, 1),
    ta.Sigmoid(),
)
model.skompiluj(ta.MSE(), ta.Adam(tempo=0.05))

dane = ta.Tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
etykiety = ta.Tensor([[0], [1], [1], [0]])
historia = model.trenuj(dane, etykiety, epoki=500, rozmiar_partii=4)
wynik = model.przewidz([[1, 0]])
print(wynik.dane)
```

## Tensor i autodiff

`Tensor` przechowuje dane jako `numpy.ndarray` typu `float32` i może śledzić
gradienty. Gradienty są używane przez parametry warstw.

```python
x = ta.Tensor([2.0], wymaga_gradientu=True)
y = x * x + 3
y.wstecz()
print(x.gradient)  # [4.]
```

Konstruktor:

```python
ta.Tensor(dane, wymaga_gradientu=False)
```

Publiczne właściwości i metody:

- `dane` – wartości tensora jako tablica NumPy.
- `gradient` – gradient po wywołaniu `wstecz()`.
- `ksztalt` – kształt danych.
- `wstecz(gradient=None)` – propagacja gradientu wstecz.
- `wyzeruj_gradient()` – usuwa zapisany gradient.
- `suma()` – suma wszystkich elementów.
- `srednia()` – średnia wszystkich elementów.

Obsługiwane są operatory `+`, `-`, `*`, `@` oraz jednoargumentowy `-`.

### Urządzenia CPU i CUDA

```python
import totaai as ta

x_cpu = ta.Tensor([1, 2])
print(x_cpu.urzadzenie)  # cpu

if ta.cuda_dostepna():
    x_gpu = ta.Tensor([1, 2], urzadzenie="cuda")
    print(x_gpu.numpy())  # kopia NumPy na CPU
```

CUDA jest opcjonalne i wymaga instalacji `pip install "totaai[cuda]"`.
Tensory używane w jednej operacji muszą znajdować się na tym samym urządzeniu.
Model przeniesiesz przez `model.na("cuda")`; jego predykcje automatycznie
umieszczają zwykłe listy wejściowe na urządzeniu modelu.

## Warstwy

Każda warstwa może być wywołana jako funkcja (`warstwa(x)`) albo przez
`przepusc(x)`. Metoda `parametry()` zwraca trenowalne tensory.

### `WarstwaLiniowa(wejscia, wyjscia)`

Warstwa gęsta wykonująca `x @ wagi + bias`.

```python
warstwa = ta.WarstwaLiniowa(784, 128)
```

Udostępnia `wagi`, `bias`, `przepusc(x)` i `parametry()`.

### `ReLU()`

Zwraca `max(0, x)`. Jest typową aktywacją dla ukrytych warstw sieci.

### `Sigmoid()`

Zwraca wartości z zakresu `(0, 1)`. Przydatna m.in. w wyjściu klasyfikacji
binarnej.

### `Softmax()`

Normalizuje ostatni wymiar do rozkładu prawdopodobieństwa. Przydatna przy
klasyfikacji wieloklasowej.

### `Tanh()` i `LeakyReLU(nachylenie=0.01)`

`Tanh` zwraca wartości z zakresu od `-1` do `1`. `LeakyReLU` zachowuje małe
nachylenie dla wartości ujemnych zamiast zerować je jak ReLU.

### `Dropout(prawdopodobienstwo=0.5)`

W czasie treningu losowo wyłącza część aktywacji. `Model.przewidz()` oraz
`Model.ocen()` automatycznie przełączają model w tryb ewaluacji, więc Dropout
nie wprowadza losowości do predykcji.

### Własna warstwa

```python
class Podwoj(Warstwa):
    def przepusc(self, x):
        return x * 2
```

## Funkcje straty

### `MSE()`

Średni błąd kwadratowy dla regresji:

```python
strata = ta.MSE()(przewidywane, oczekiwane)
```

### `EntropiaKrzyzowa()`

Stabilna numerycznie entropia krzyżowa dla logitów i indeksów klas:

```python
strata = ta.EntropiaKrzyzowa()(logity, klasy)
```

`klasy` powinny zawierać indeksy klas, np. `[0, 2, 1]`.

### `MAE()` i `EntropiaBinarna()`

`MAE` to średni błąd bezwzględny dla regresji, mniej wrażliwy na wartości
odstające niż MSE. `EntropiaBinarna` służy do klasyfikacji 0/1 i wymaga
wyjścia `Sigmoid()`.

## Optymalizatory

### `SGD(tempo=0.01, ped=0.0)`

Prosty spadek gradientu. `tempo` oznacza learning rate, a `ped` dodaje
regularyzację L2.

### `Adam(tempo=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8)`

Adaptacyjny optymalizator z momentami pierwszego i drugiego rzędu.

Oba optymalizatory posiadają `krok(parametry)` i
`wyzeruj_gradient(parametry)`. Zwykle są używane przez `Model.trenuj()`.

## Model

```python
model = ta.Model()
```

- `dodaj(*warstwy)` – dodaje warstwy i zwraca model, więc można łączyć wywołania.
- `skompiluj(strata, optymalizator)` – konfiguruje trening.
- `przepusc(x)` – propagacja w przód dla tensora.
- `przewidz(x)` – predykcja dla tensora, listy lub tablicy NumPy.
- `parametry()` – wszystkie parametry trenowalne.
- `podsumowanie()` – wypisuje architekturę i liczbę parametrów.
- `ocen(dane, etykiety)` – zwraca średnią wartość funkcji straty.
- `zapisz(sciezka)` – zapisuje model wraz z wagami.
- `Model.wczytaj(sciezka)` – odczytuje zapisany model.

### Trening

```python
historia = model.trenuj(
    dane,
    etykiety,
    epoki=10,
    rozmiar_partii=32,
    tasuj=True,
    pokazuj_postep=True,
)
```

Metoda zwraca listę strat, po jednej wartości na epokę. `rozmiar_partii=None`
oznacza trening pełnym zbiorem. `tasuj` domyślnie miesza przykłady na początku
każdej epoki.

### Walidacja i historia

```python
model.trenuj(
    dane_treningowe, etykiety_treningowe,
    epoki=20,
    walidacja=(dane_walidacyjne, etykiety_walidacyjne),
)
print(model.historia["wal_strata"])
```

`historia` zawiera stratę treningową i walidacyjną z każdej epoki.

## Narzędzia danych i metryki

```python
x_trening, x_test, y_trening, y_test = ta.podziel_dane(
    dane, etykiety, udzial_testowy=0.2, ziarno=42
)
wyniki = model.przewidz(x_test)
print(ta.dokladnosc(wyniki, y_test))
print(ta.blad_sredni_bezwzgledny(wyniki, y_test))
```

- `podziel_dane()` zwraca dane i etykiety treningowe/testowe.
- `dokladnosc()` obsługuje klasyfikację binarną i wieloklasową.
- `blad_sredni_bezwzgledny()` zwraca MAE jako zwykłą liczbę.

## Zapis i import modelu

```python
model.zapisz("model.tota")
odtworzony = ta.Model.wczytaj("model.tota")
print(odtworzony.przewidz([[1, 0]]).dane)
```

Format jest przeznaczony dla zaufanych plików lokalnych. Nie wczytuj plików
modeli pochodzących z nieznanego źródła.

## Aktualny zakres

Obecna wersja nie zawiera jeszcze CNN, RNN, Transformerów, autoenkoderów,
przetwarzania obrazów/audio, GPU ani gotowych loaderów danych. Te elementy są
planowane jako osobne moduły, aby nie komplikować stabilnego rdzenia.

## Tutoriale

### 1. XOR: pierwszy model

```python
model = ta.Model().dodaj(
    ta.WarstwaLiniowa(2, 8), ta.Sigmoid(),
    ta.WarstwaLiniowa(8, 1), ta.Sigmoid(),
)
model.skompiluj(ta.MSE(), ta.Adam(tempo=0.05))
model.trenuj(
    [[0, 0], [0, 1], [1, 0], [1, 1]],
    [[0], [1], [1], [0]],
    epoki=500,
    pokazuj_postep=False,
)
print(model.przewidz([[1, 0]]).dane)
```

### 2. Gradient pojedynczego tensora

```python
x = ta.Tensor([2.0], wymaga_gradientu=True)
y = x * x + 3
assert x.gradient[0] == 4.0
```

### 3. Klasyfikacja trzech klas

```python
model = ta.Model().dodaj(
    ta.WarstwaLiniowa(4, 16), ta.ReLU(),
    ta.WarstwaLiniowa(16, 3), ta.Softmax(),
)
model.skompiluj(ta.EntropiaKrzyzowa(), ta.Adam())
model.trenuj(dane, klasy, epoki=50, rozmiar_partii=32)
klasy_przewidziane = model.przewidz(dane_testowe).dane.argmax(axis=-1)
```

### 4. Partie i walidacja

```python
model.trenuj(
    dane_treningowe, etykiety_treningowe,
    epoki=25, rozmiar_partii=32, tasuj=True,
)
blad = model.ocen(dane_walidacyjne, etykiety_walidacyjne)
print(f"Blad walidacji: {blad:.4f}")
model.podsumowanie()
```

### 5. Zapis i odczyt

```python
model.zapisz("model.tota")
odtworzony = ta.Model.wczytaj("model.tota")
wynik = odtworzony.przewidz(dane_testowe)
```
