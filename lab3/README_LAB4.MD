# README – Lab 4

## Podsumowanie zadania
W tym laboratorium zajmowałem się optymalizacją hiperparametrów modelu sieci neuronowej. Na początku zapisałem wynik modelu bazowego (baseline), a później sprawdzałem, czy uda się go poprawić tuningiem.

Testowane były takie parametry jak:
- liczba neuronów w ukrytej warstwie,
- funkcja aktywacji,
- tempo uczenia (learning rate),
- dropout.

Do strojenia użyłem prostego tuningu (Keras Tuner – RandomSearch), który sprawdzał różne ustawienia i wybierał najlepszy model.

## Wyniki
- **Wynik baseline:** 0.8888889
- **Wynik po tuningu:** 0.8888889

W moim przypadku dokładność nie wzrosła, ale tuning potwierdził, że model, który miałem wcześniej, był już całkiem dobry.

## Macierz pomyłek
Macierz pomyłek wygląda tak:

 [[12  0  0]

 [ 0 14  0]

 [ 0  0 10]]


Oznacza to, że model poprawnie sklasyfikował wszystkie próbki w obu klasach – nie było żadnych pomyłek.

## Podsumowanie modelu (`model.summary()`)
Warstwa (typ)        Output Shape   Param #

normalization       (None, 13)     27      
dense               (None, 160)    2240    
dropout             (None, 160)    0       
dense_1             (None, 3)      483     

**Łącznie parametrów:** 2750  
**Trenowalnych:** 2723

## Metryki
- **Accuracy (dokładność):** ~0.889  
- **Macierz pomyłek:** brak błędnych klasyfikacji

## Wnioski
Tuning hiperparametrów pozwolił sprawdzić różne warianty modelu, ale najlepszy wynik okazał się taki sam jak model bazowy. Mimo to nauczyłem się, jak korzystać z Keras Tunera i jak oddzielić tworzenie modelu do osobnej funkcji. Model działa stabilnie i dobrze radzi sobie z klasyfikacją danych.
