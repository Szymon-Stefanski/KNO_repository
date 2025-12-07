## Podsumowanie wyników

Model CNN, który trenowałem na zbiorze Fashion-MNIST, osiągnął:

- 91.7% accuracy na zbiorze treningowym,
- 90.8% accuracy na zbiorze walidacyjnym,
- trening trwał 10 epok.

Różnica między train a val accuracy jest niewielka, więc nie widać przeuczenia. Strata (loss) systematycznie 
spadała w kolejnych epokach, co pokazuje, że model uczył się prawidłowo.

Wszystkie metryki zostały zapisane do pliku metrics.json, a macierz pomyłek do confusion_matrix.png.

Model został również przetestowany na własnym zdjęciu pobranym z internetu. Po odpowiednim przetworzeniu obrazu 
(skalowanie do 28×28, konwersja na grayscale, negatyw) model poprawnie rozpoznał element garderoby jako T-shirt/top z 
pewnością około 89%, co jest dobrym wynikiem jak na obraz spoza zbioru treningowego.

Podsumowując: model działa poprawnie, osiąga oczekiwane rezultaty na Fashion-MNIST i radzi sobie z klasyfikacją nowych 
obrazów.
