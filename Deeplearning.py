import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

# Defining the data directories
train_dir = r"D:\Mohammad\PyMo\ROI\Data set\Brain Tumor Dataset\Training"
test_dir = r"D:\Mohammad\PyMo\ROI\Data set\Brain Tumor Dataset\Testing"

# Defining the data generators for training and testing data
#ImageDatenGeneratoe 👉 Das ist eine TensorFlow-Klasse, die:
# ✔ Bilder lädt
# ✔ Bilder transformiert
# ✔ Bilder fürs Training vorbereitet
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150), # Resizing the images to 150x150 pixels| alle Bilder gleich groß sein müssen
    #CNN kann nur gleiche Größen verarbeiten
    batch_size=32,#Das Modell sieht immer 32 Bilder gleichzeitig,
    # Warum nicht alles auf einmal?
    # zu viel Speicher ❌
    # langsamer ❌
    # instabil ❌
    class_mode='categorical' # categorical cross-entropy as loss function
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

# 🧠 Einfach erklärt

# 👉 Pixelwert = „Wie hell ist der Punkt?“
# 👉 Pixelgröße = „Wie viele Punkte hat das Bild?

# Analogie | Stell dir ein Mosaik vor:
# Pixelgröße = wie viele Steine du hast
# Pixelwert = welche Farbe jeder Stein hat

# 🧠 MERKSATZ
# 👉 Pixelwert = Farbe/Helligkeit
# 👉 Pixelgröße = Auflösung/Dimension


# Anzahl Bilder pro Klasse zählen
import numpy as np



def print_class_distribution(generator, name):
    print(f"\n📂 {name}")

    classes = generator.class_indices
    counts = np.bincount(generator.classes)

    for class_name, index in classes.items():
        print(f"{class_name}: {counts[index]} Bilder")


# Training
print_class_distribution(train_generator, "Training Data")

# Testing
print_class_distribution(test_generator, "Testing Data")


# Defining the CNN model
# CNN interessiert sich nicht für jedes einzelne Pixel, sondern für:
# ✔ Kanten
# ✔ Formen
# ✔ Muster
# ✔ Strukturen


#Modelerstelle<>

# Ein Filter ist ein kleines „Suchmuster“, das über das Bild läuft.

# 👉 Er sucht bestimmte Dinge im Bild:

# Kanten
# Linien
# Muste

# Ein Filter kann erkennen:

# ✔ vertikale Linien
# ✔ horizontale Linien
# ✔ Texturen

# Original Bild (150,150) es werden 32 Filter gleichzeitig eingesetzt mit 32 Filter = 32 verschiedene Perspektiven und
#  suchen einen bestimmten Muster in dem Orginalbild, das Bild wird 
# 3x3pixel gescant , am ende entseht für jeden Filter einen Musterbild
#es werden 32 neue Bilder (Feature Maps)

#       ↓
# [Filter 1] → Kantenbild
# [Filter 2] → Linienbild
# [Filter 3] → Musterbild
...
#[Filter 32] → anderes Feature

# Convolution = der Prozess, bei dem ein Filter über das Bild „gleitet“
# 👉 Filter = Muster-Erkenner
# 👉 Convolution = Scan-Prozess

######Ein CNN verarbeitet nicht nur ein einzelnes Bild, sondern immer einen Batch (z. B. 32 Bilder gleichzeitig). Jedes dieser Bilder wird einzeln durch das Netzwerk geleitet.

# Im ersten Convolution-Layer wird jedes Bild mit 32 verschiedenen Filtern
#  (3×3 Kernel) bearbeitet. Diese Filter scannen das Bild lokal und 
# erkennen einfache Muster wie Kanten, Ecken oder Texturen. Das Ergebnis
#  sind sogenannte Feature Maps – das sind keine echten Bilder, sondern
#  mathematische Darstellungen der erkannten Muster.

# Diese Feature Maps werden dann nicht gespeichert, sondern direkt als Eingabe
#  für die nächste Layer verwendet.

# In der zweiten Convolution-Layer (z. B. mit 64 Filtern) wird also 
# nicht wieder das Originalbild verwendet, sondern die bereits verarbeiteten 
# Feature Maps aus der vorherigen Schicht. Dadurch erkennt 
# das Netzwerk immer komplexere Muster (z. B. Formen oder Tumor-Strukturen).

# Zwischen den Convolution-Layern reduziert MaxPooling
# die Größe der Daten, damit nur die wichtigsten Informationen 
# erhalten bleiben.

# Am Ende werden alle extrahierten Merkmale in einer Dense-Schicht 
# zusammengeführt und die Softmax-Schicht gibt die Prediction aus,
#  also die Wahrscheinlichkeit für jede Klasse (z. B. Tumorarten)

model = tf.keras.models.Sequential([#Ich baue ein Modell Schicht für Schicht hintereinander.
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),#Erste Convolution Layer
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),#Zweite Conv Layer
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),#drite Conv Layer
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),#Bild wird von 2D → 1D umgewandelt damit es in Dense Layer passt
    tf.keras.layers.Dense(128, activation='relu'), #👉 Jetzt „denkt“ das Modell ✔ kombiniert alle Features ✔ trifft Entscheidungen
    tf.keras.layers.Dropout(0.5),#50% der Neuronen werden zufällig ausgeschalte ✔ verhindert Overfitting ✔ macht Modell robuster
    tf.keras.layers.Dense(3, activation='softmax')#Prediction |Output Layer|finale Entscheidung|gibt Wahrscheinlichkeiten glioma: 0.10 meningioma: 0.80 no tumor: 0.10|höchste gewinnt
])

# # Compiling the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# 🧠 model.compile(...) bedeutet:👉 „Ich stelle das Lernsystem ein“

#loss = Fehlerfunktion|Das bedeutet:misst, wie falsch das Modell ist und vergleicht Prediction vs. echtes Labe

#optimizer = Lernalgorithmus|Das ist der „Trainer“ Er entscheidet: wie stark Gewichte verändert werden 
# in welche Richtung gelernt wird

#metrics=['accuracy'] nur A|nzeige|zur Kontrolle|wie viele Bilder richtig erkannt wurden

# Training the model
history = model.fit(
    train_generator,
    epochs=20,#10 Epochs = das Modell sieht ALLE Bilder 10-mal in Training Dataset
    validation_data=test_generator # Modell sieht neue Bilder die es NICHT lernen darf aud dem TestingDataset
)


#Evaluating the model on the test data
# 🧠 model.evaluate(...) bedeutet:

# 👉 „Teste das fertige Modell mit neuen Daten“
test_loss, test_acc = model.evaluate(test_generator, verbose=2)
#Evaluate=exam
# will ich Detail (verbose=1)?
# oder nur Ergebnis (verbose=2)?
# oder gar nichts (verbose=0)?

print('Test accuracy:', test_acc)


#saving the model #.Keras is KI Gehirn (Modell) 🧠
from tensorflow import keras
model.save(r"D:\Mohammad\PyMo\ROI\Data set\Brain Tumor Dataset\BrainTumorProject.keras")
#model.save# 👉 Du speicherst dein fertig trainiertes Gehirn

# Das beinhaltet:

# ✔ Architektur (CNN Aufbau)
# ✔ gelernte Gewichte (Filter)
# ✔ Bias-Werte
# ✔ Trainingsstand

my_model =tf.keras.models.load_model(r"D:\Mohammad\PyMo\ROI\Data set\Brain Tumor Dataset\BrainTumorProject.keras")

#Load_model# 👉 Du holst das gespeicherte Gehirn zurück
# ohne neu zu trainieren

########Code Visuell machen nicht nur Zahlen?#######

# Getting 10 random images from the test set
images, labels = next(test_generator)
indices = np.random.choice(len(images), 10, replace=False)# np=numpy
images = images[indices]
labels = labels[indices]

# 👉 next() = holt einen Batch
# 👉 np.random.choice() = wählt zufällig
# 👉 images[indices] = filtert Auswah
#Was macht flow_from_directory(...)
# Dieser Generator baut intern:
# Liste:
# img1.jpg → glioma →[1,0,0]
# img2.jpg → no tumor → [0,0,1]
# img3.jpg → meningioma →[0,1,0]

# Bild|X = [img1, img2, img3]
# Label|Y = [[1,0,0], [0,0,1], [0,1,0]]

# Deswegen es sollte den Datenset so gespeichert in drei unterordner(Glioma, Meniogoma, no tumor)


# # Making predictions on the 10 images
predictions = my_model.predict(images)

# # Decoding the one-hot encoded labels and predictions
label_names = test_generator.class_indices
# ausgabe sind
# {
#   'glioma': 0,
#   'meningioma': 1,
#   'no_tumor': 2
# }

label_names = dict((v,k) for k,v in label_names.items())
# Umwandlung von indexedict gibt die werte in label_names wieder als Dictionara (value, Key) zurück.
# items() um beiden werte zu holenn --> output
# ('glioma', 0)
# ('meningioma', 1)
# 
true_labels = [label_names[np.argmax(label)] for label in labels]
# label = [0, 1, 0], argmax sucht in diesem label den größten wert also 1 und das enstrpricht "Meniogioma"
predicted_labels = [label_names[np.argmax(pred)] for pred in predictions]

# # Displaying the 10 images with their true and predicted labels
import matplotlib.pyplot as plt

fig, axs = plt.subplots(2, 5, figsize=(15, 6)) # inches 15,6 
for i, ax in enumerate(axs.flat):# 
    #ohne flat
    # Zeile 1: ax1 ax2 ax3 ax4 ax5
    # Zeile 2: ax6 ax7 ax8 ax9 ax10
    #mit flat
    # ax1, ax2, ax3, ..., ax10 ❗ aber NICHT visuell im Plot! 
    ax.imshow(images[i]) #images[i] # put image an dieser Position |ax.imshow(images[i])„Zeige Bild Nr. i im Feld ax“
    ax.set(title=f'True: {true_labels[i]} \n Predicted: {predicted_labels[i]}')
    ax.axis('off')  

num_epochs = len(history.history['loss'])
acc = test_acc

plt.savefig(
    rf"D:\Mohammad\PyMo\ROI\Data set\Brain Tumor Dataset\BrainTumor_epoch{num_epochs}_acc{acc:.2f}.png",
    dpi=300,
    bbox_inches='tight'
)


plt.show()
