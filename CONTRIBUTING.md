# Współtworzenie TotaAI

## Przygotowanie

```bash
git clone https://github.com/totatomasz13-web/TotaAI.git
cd TotaAI
python -m venv .venv
.venv/Scripts/activate  # Windows
pip install -e ".[dev]"
```

## Przed pull requestem

Uruchom testy i build strony:

```bash
pytest
cd website
npm ci
npm run build
```

Zmiany w `website/` wymagają przeglądu właściciela strony. Nie dodawaj
sekretów, kluczy API ani plików `.env` do repozytorium.

## Zasady

- Używaj polskich nazw publicznego API.
- Dodaj test do każdej nowej funkcji rdzenia.
- Aktualizuj `docs/API.md` razem ze zmianą API.
- Zachowuj małe, opisowe commity w stylu Conventional Commits.
- Nie omijaj kontroli CI ani ochrony gałęzi `main`.
