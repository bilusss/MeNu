import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Flatten, Dense, Input
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
import matplotlib.pyplot as plt
import numpy as np
import time

'''
po kazdym cyklu adaptacji wag sieci z uzyciem wszystkich danych
uczacych (po kazdej tzw. epoce): 1) oblicza sie wartosc funkcji kosztu/błedu
dla danych uczacych i walidacyjnych, 2) oblicza sie dokładnosc rozpoznawania
dla danych uczacych i walidacyjnych, 3) obserwuje sie otrzymywane krzywe i
zatrzymuje proces uczenia w odpowiednim momencie - ma to zapobiec efektowi
przeuczenia sieci, czyli nauczeniu sie przykładów ze zbioru treningowego na
pamiec i utracenia zdolnosci podejmowania poprawnych decyzji dla nowych
(nieznanych wczesniej danych)

L2 uzywa sie by uniknac overfitting, czyli przeuczenia sie modelu.
wzor na L2 np. L2 = SIGMA(n->inf) (Wn)^2

klasyfikator multiklasowy ma wyjscia rodzaju: multi-class (ONEvsALL),
tak jak w rozpoznawaniu liczb z zakresu 0-9, uzycie techniki soft-max!!!

Liczba warstw ukrytych (hidden_layers).
Liczba neuronów w każdej warstwie (neurons_per_layer).
Współczynnik regularyzacji L2 (l2_reg), który zapobiega przeuczeniu (overfittingowi).
Inicjalizacja wag (init), która wpływa na startowy rozkład wartości wag sieci neuronowej.


@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


Plot 1 (Hidden Layers: 2, Neurons per Layer: 128, L2 reg: 0.0, init: glorot_uniform)
Accuracy Plot: Dokładność treningowa szybko rośnie, ale dokładność walidacyjna stabilizuje się wcześniej,
co wskazuje na ryzyko przeuczenia (brak regularyzacji).

Loss Plot: Straty treningowe maleją szybko, ale różnica między stratą treningową a walidacyjną się powiększa,
co oznacza brak ochrony przed overfittingiem.


Plot 2 (Hidden Layers: 2, Neurons per Layer: 128, L2 reg: 0.01, init: glorot_uniform)
Accuracy Plot: Dokładność rośnie wolniej, ale krzywe treningowe i walidacyjne są bardziej zbliżone,
co świadczy o lepszej generalizacji dzięki regularyzacji.

Loss Plot: Straty walidacyjne zmniejszają się równomiernie ze stratami treningowymi,
co pokazuje skuteczność regularyzacji w redukcji przeuczenia.


Plot 3 (Hidden Layers: 2, Neurons per Layer: 128, L2 reg: 0.1, init: glorot_uniform)
Accuracy Plot: Krzywe treningowe i walidacyjne są bardzo zbliżone, ale dokładność jest niższa,
co wskazuje na zbyt dużą regularyzację i niedouczenie.

Loss Plot: Straty są wyższe zarówno dla danych treningowych, jak i walidacyjnych,
ponieważ model ma ograniczone możliwości uczenia się.


Plot 4 (Hidden Layers: 2, Neurons per Layer: 128, L2 reg: 0.0)
Accuracy Plot: Krzywa dokładności dla danych treningowych wzrasta z każdą epoką,
ale dokładność walidacji zaczyna się stabilizować, co wskazuje na zbliżanie się do limitu modelu.
Nie użyto regularyzacji, co może skutkować overfittingiem w dalszych epokach.

Loss Plot: Straty treningowe maleją konsekwentnie, ale różnica między stratą treningową i walidacyjną
może się powiększać bez regularyzacji, co potwierdza brak ochrony przed overfittingiem.


Plot 5 (Hidden Layers: 2, Neurons per Layer: 128, L2 reg: 0.01)
Accuracy Plot: Krzywa dokładności treningowej rośnie wolniej w porównaniu do poprzedniego modelu,
ale dokładność walidacyjna jest bardziej stabilna. Dodanie regularyzacji L2 pomogło zapobiec przeuczeniu.

Loss Plot: Straty walidacyjne zmniejszają się bardziej proporcjonalnie do strat treningowych,
co pokazuje efekt regularyzacji (karania dużych wag), co poprawia generalizację.


Plot 6 (Hidden Layers: 2, Neurons per Layer: 128, init: glorot_uniform)
Accuracy Plot: Wyniki są stabilne od początku, ponieważ inicjalizacja „Glorot Uniform” (Xavier Initialization)
zapewnia, że wagi początkowe są dobrze zbalansowane. Model uczy się szybko i dobrze radzi sobie na danych walidacyjnych.

Loss Plot: Obie krzywe strat (treningowa i walidacyjna) są gładkie i spójne, co wskazuje, że inicjalizacja wag
przyczyniła się do efektywnego trenowania.


@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
'''

# MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalizacja danych
X_train = X_train / 255.0
X_test = X_test / 255.0

# Funkcja do wyświetlania przykładowych obrazów
# def plot_sample_images(X, y):
#     plt.figure(figsize=(10, 10))
#     for i in range(16):
#         plt.subplot(4, 4, i + 1)
#         plt.imshow(X[i], cmap='gray')
#         plt.title(f"Label: {y[i]}")
#         plt.axis('off')
#     plt.show()
#
# plot_sample_images(X_train, y_train)

# Funkcja do rysowania historii uczenia z parametrami
def plot_training_history_with_params(history, params):
    """
    Funkcja rysująca historię uczenia modelu z wyświetlaniem użytych parametrów.
    """
    plt.figure(figsize=(12, 6))

    # Wyświetlanie parametrów jako tekstu nad wykresami
    params_text = f"HiddenLayers:{params['hidden_layers']},NeuronsPerLayer:{params['neurons_per_layer']}," \
                  f"epochs:{params['epochs']},L2reg:{params['l2_reg']:.4f}," \
                  f"init:{params['init']}"

    # Wykres dokładności
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Model Accuracy')
    plt.text(0.5, 1.1, params_text, horizontalalignment='center', verticalalignment='center',
             transform=plt.gca().transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

    # Wykres strat
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Model Loss')
    plt.text(0.5, 1.1, params_text, horizontalalignment='center', verticalalignment='center',
             transform=plt.gca().transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

    plt.tight_layout()
    plt.show()

# Funkcja do tworzenia i trenowania modelu z różnymi konfiguracjami
def train_and_evaluate_model_with_plot(hidden_layers, neurons_per_layer, epochs, l2_reg=0.0, init='glorot_uniform'):
    # Tworzenie modelu
    model = Sequential()
    model.add(Input(shape=(28, 28)))
    model.add(Flatten())

    for _ in range(hidden_layers):
        model.add(Dense(neurons_per_layer, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                        kernel_initializer=init))

    model.add(Dense(10, activation='softmax'))

    # Kompilacja modelu
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    # Trenowanie modelu
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=32,
        verbose=2
    )

    params = {
        "hidden_layers": hidden_layers,
        "neurons_per_layer": neurons_per_layer,
        "epochs": epochs,
        "l2_reg": l2_reg,
        "init": init
    }

    plot_training_history_with_params(history, params)

    # Walidacja modelu
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Hidden Layers: {hidden_layers}, Neurons per Layer: {neurons_per_layer}, Epochs: {epochs}")
    print(f"Test Accuracy: {test_accuracy:.4f}, Test Loss: {test_loss:.4f}")

    return model, history, test_accuracy

# Analiza współczynnika regularyzacji
for l2_reg in [0.0, 0.01, 0.1]:
    train_and_evaluate_model_with_plot(2, 128, epochs=10, l2_reg=l2_reg)

# Analiza inicjalizacji wag
for init in ['glorot_uniform', 'he_normal', 'random_normal']:
    train_and_evaluate_model_with_plot(2, 128, epochs=10, init=init)
