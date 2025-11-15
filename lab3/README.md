# Opis wyników

W projekcie porównałem dwa modele sieci neuronowych typu Sequential, które miały za zadanie klasyfikować wina na 3 kategorie (zbiór Wine z UCI).

## 1. Modele

### Model 1
- Dwie warstwy ukryte: 64 i 32 neurony
- Aktywacja ReLU
- Wyjście: Softmax
- Jest to prostszy model, ma mniej parametrów

### Model 2
- Dwie większe warstwy: 128 i 64 neurony
- Aktywacja ELU
- Inicjalizacja He Normal
- Dodatkowo warstwa Dropout 0.3
- Model bardziej rozbudowany od pierwszego

## 2. Wyniki trenowania

Oba modele uczyły się poprawnie i szybko zbiegały.  
Wyniki na zbiorze testowym były następujące (mogą się minimalnie różnić przy ponownym uruchomieniu):

- **Model 1:** ~97% accuracy  
- **Model 2:** ~98% accuracy

Na wykresach z matplotlib i w TensorBoard było widać, że Model 2 ma trochę niższą stratę walidacyjną i ogólnie radzi sobie minimalnie lepiej.

## 3. Wnioski

- **Lepszym modelem okazał się Model 2**, prawdopodobnie dlatego, że ma więcej neuronów i dropout, co pomaga uniknąć przeuczenia.
- Różnica nie jest duża, ponieważ zbiór danych jest dość mały i łatwy, ale mimo wszystko Model 2 generalizuje trochę lepiej.

## 4. Predykcja

Po treningu najlepszy model jest zapisywany do pliku `best_model.h5`, a standaryzator w `scaler.pkl`.

Można wykonać predykcję z linii komend, podając 13 cech wina, np.:

```bash
python wine_train_and_predict.py --predict --features 13.2 2.1 2.4 18.0 100 2.3 2.0 0.3 1.7 5.4 1.04 3.40 1050
