# Zgłaszanie problemów bezpieczeństwa

Nie publikuj podatności w publicznym issue. Zgłoś ją prywatnie właścicielowi
repozytorium przez funkcję **Private vulnerability reporting** na GitHubie.

Nie wczytuj modeli z nieznanego źródła: `Model.wczytaj()` używa mechanizmu
pickle, który może wykonywać kod podczas deserializacji.
