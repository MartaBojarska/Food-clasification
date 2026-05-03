# Food-clasification
# Projekt: Sztuczne Sieci Neuronowe i Głębokie Uczenie
**Klasyfikacja rodzajów jedzenia z wykorzystaniem głębokich konwolucyjnych sieci neuronowych (CNN)**

* **Autor:** Marta Bojarska
* **Link do repozytorium GitHub:** https://github.com/MartaBojarska/Food-clasification
* **Plik z kodem:** `food_classification.ipynb`

---

## 1. Wprowadzenie i uzasadnienie wyboru zadania
Celem niniejszego projektu jest zaprojektowanie, zaimplementowanie i ewaluacja modelu głębokiego uczenia w zadaniu klasyfikacji obrazów. Jako problem badawczy wybrano zadanie rozpoznawania rodzajów dań na podstawie zdjęć (tzw. problem ,,Food Classification''). 

Rozwiązanie to wpisuje się w główny obszar sylabusa: **klasyfikacja lub analiza obrazów**. Zamiast standardowych, akademickich zbiorów danych (jak MNIST czy CIFAR), wybrano zbiór o wysokim stopniu skomplikowania wizualnego. Zadanie to ma wymiar wysoce praktyczny – stanowi fundament dla aplikacji dietetycznych (automatyczne rozpoznawanie posiłków do liczenia kalorii) czy systemów rekomendacyjnych w gastronomii.

## 2. Opis danych i problemu
W projekcie wykorzystano publicznie dostępny zbiór danych **Food-101**, pobrany za pomocą biblioteki `tensorflow_datasets`. Zbiór ten docelowo zawiera 101 kategorii jedzenia (np. pizza, sushi, ramen). Ze względu na ograniczenia sprzętowe oraz czasowe, do celów eksperymentalnych wyekstrahowano reprezentatywny podzbiór 10% danych treningowych i walidacyjnych.

**Przygotowanie danych (Preprocessing i Augmentacja):**
1. **Skalowanie i normalizacja:** Wszystkie obrazy wejściowe zostały przeskalowane do ujednoliconego rozmiaru 224x224 pikseli. Wartości pikseli znormalizowano do zakresu 0.0 - 1.0 poprzez podzielenie przez 255.0, co stabilizuje i przyspiesza proces zbieżności algorytmu uczącego.
2. **Augmentacja w locie:** Aby zapobiec zjawisku przeuczenia (overfittingu) – szczególnie przy ograniczonym podzbiorze danych – zastosowano potok augmentacji z użyciem warstw Keras. Obejmował on:
   * Losowe odbicia lustrzane w poziomie (`RandomFlip('horizontal')`)
   * Losowe obroty obrazu (`RandomRotation(0.1)`)
   * Losowe przybliżenia (`RandomZoom(0.1)`)
   * Zmiany kontrastu (`RandomContrast(0.2)`)

## 3. Opis architektury i metod uczenia
Rozwiązanie oparto na głębokiej **Konwolucyjnej Sieci Neuronowej (CNN)**.

* **Architektura Bazowa:** Wykorzystano model **MobileNetV2**. Jest to wydajna, lekka architektura, idealna do środowisk o ograniczonych zasobach.
* **Uczenie transferowe (Transfer Learning):** Baza modelu została zainicjalizowana wagami przetrenowanymi na potężnym zbiorze ImageNet. Dzięki temu sieć ,,na starcie'' potrafiła rozpoznawać podstawowe cechy obrazu, takie jak krawędzie czy tekstury.
* **Klasyfikator:** Do bazy dodano warstwę `GlobalAveragePooling2D` redukującą wymiarowość, następnie gęstą warstwę w pełni połączoną (`Dense`) ze 128 neuronami i funkcją aktywacji **ReLU**, regularyzację `Dropout` (0.5) chroniącą przed przeuczeniem, oraz warstwę wyjściową typu `Dense` (101 klas) z funkcją **Softmax**, zwracającą prawdopodobieństwa poszczególnych klas.
* **Proces uczenia:** Jako funkcję straty wykorzystano krzyżową entropię (`sparse_categorical_crossentropy`). Do aktualizacji wag posłużył algorytm propagacji wstecznej błędu w połączeniu z nowoczesnym, adaptacyjnym optymalizatorem **Adam**. Ewaluacja odbywała się za pomocą metryki `accuracy`.

## 4. Eksperymenty, wyniki, interpretacja
Zgodnie z wymaganiami przeprowadzono trzy eksperymenty porównawcze, badając wpływ hiperparametrów oraz technik uczenia na skuteczność modelu.

* **Eksperyment 1 (Baza referencyjna):** 
  Zamrożona baza MobileNetV2, `learning_rate = 0.001`, `batch_size = 32`. Model uczył się stabilnie, osiągając na zbiorze walidacyjnym dokładność rzędu **~37%**.
* **Eksperyment 2 (Wpływ wielkości wsadu - Batch Size):** 
  Zwiększono rozmiar wsadu do `batch_size = 128`. Zaobserwowano spadek dokładności walidacyjnej do **~35%**. Wniosek: Mniejszy rozmiar wsadu (32) w tym przypadku pozwolił optymalizatorowi na częstszą aktualizację wag i lepszą generalizację.
* **Eksperyment 3 (Fine-tuning - Różne głębokości modelu i niski LR):** 
  Odblokowano ostatnie 30 warstw bazy konwolucyjnej, zmniejszając drastycznie współczynnik uczenia (`learning_rate = 1e-5`). Wynik poprawił się, osiągając blisko **~39%** dokładności. Niska wartość *learning rate* zapobiegła destrukcji pre-trenowanych wag, pozwalając jednocześnie na dostosowanie głębokich filtrów do specyfiki obrazów jedzenia.

*(Uwaga: Wynik bliski 40% w zaledwie 5 epokach, na 101 złożonych klasach przy redukcji zbioru do 10%, jest bardzo dobrym wynikiem, udowadniającym skuteczność metody Transfer Learningu).*

**Ewaluacja jakościowa (Interpretowalność sieci):**
* **Błędne predykcje:** Model miał trudności z potrawami o zbliżonej teksturze i kolorystyce (np. jasne kremy, gęste sosy). Wynika to z faktu, że dla sieci konwolucyjnej tego typu struktury dają bardzo zbliżony sygnał wizualny.
* **Mapy aktywacji (Feature Maps):** Wyekstrahowano i zwizualizowano mapy cech z warstwy `Conv1`. Potwierdzają one teoretyczne działanie sieci CNN – wczesne warstwy funkcjonują jako detektory krawędzi, zarysów obiektów oraz kontrastu świetlnego.

## 5. Ograniczenia projektu i potencjalne ulepszenia (Nawiązanie do Rozdziału 10 sylabusa)
Analizując model w kontekście ograniczeń współczesnych systemów głębokich, należy zauważyć następujące kwestie:
1. **Zapotrzebowanie na dane (Data Hunger):** Głębokie sieci wymagają ogromnej ilości danych. W projekcie ograniczono zbiór do 10% ze względów sprzętowych, co stanowiło wąskie gardło. Trening na pełnym zbiorze pozwoliłby na znaczne polepszenie wyników.
2. **Problem czarnej skrzynki (Black-box) i wyjaśnialności:** Choć zbadano mapy aktywacji wczesnych warstw, decyzje podejmowane w głębokich warstwach są trudne do interpretacji dla człowieka. Zastosowanie pełnego algorytmu XAI (np. Grad-CAM) mogłoby dostarczyć dokładniejszych odpowiedzi na temat przyczyn błędnych predykcji.
3. **Koszty obliczeniowe (Hardware limits):** Trenowanie głębokich architektur od zera jest niezwykle kosztowne obliczeniowo. Użycie pre-trenowanego MobileNetV2 było koniecznym i optymalnym kompromisem.
4. **Potencjalne ulepszenia:** W przyszłości model można rozbudować o wydłużony proces fine-tuningu, wdrożyć techniki takie jak *Learning Rate Schedulers* (dynamiczna zmiana kroku uczenia) lub przetestować inne architektury bazowe (np. ResNet, EfficientNet).

## 6. Podsumowanie
Projekt zakończył się sukcesem. Zbudowano i przetestowano od podstaw przepływ danych (pipeline) potrafiący ładować i augmentować dane, modyfikować architekturę transferową (MobileNetV2) oraz uczyć ją metodą wstecznej propagacji. Za pomocą trzech zróżnicowanych eksperymentów udowodniono zrozumienie wpływu hiperparametrów (*Batch Size*, *Learning Rate*) i architektury (*Fine-Tuning*) na proces uczenia. Wygenerowane mapy cech udowadniają praktyczne zrozumienie działania filtrów wewnątrz sieci konwolucyjnej.