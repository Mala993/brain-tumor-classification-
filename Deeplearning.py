import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

# Defining the data directories
train_dir = r"D:\Mohammad\PyMo\ROI\Data set\Brain Tumor Dataset\Training"
test_dir = r"D:\Mohammad\PyMo\ROI\Data set\Brain Tumor Dataset\Testing"


train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150), 
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

# make number of class visible
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

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'), 
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(3, activation='softmax')
])

# Compiling the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



# Training the model
history = model.fit(
    train_generator,
    epochs=20,# 1,10,20
    validation_data=test_generator 
)


# Evaluating the model on the test data

test_loss, test_acc = model.evaluate(test_generator, verbose=2)

print('Test accuracy:', test_acc)


#saving the model 
from tensorflow import keras
model.save(r"D:\Mohammad\PyMo\ROI\Data set\Brain Tumor Dataset\BrainTumorProject.keras")

my_model =tf.keras.models.load_model(r"D:\Mohammad\PyMo\ROI\Data set\Brain Tumor Dataset\BrainTumorProject.keras")


# Getting 10 random images from the test set
images, labels = next(test_generator)
indices = np.random.choice(len(images), 10, replace=False)
images = images[indices]
labels = labels[indices]




#  Making predictions on the 10 images
predictions = my_model.predict(images)

#  Decoding the one-hot encoded labels and predictions
label_names = test_generator.class_indices

label_names = dict((v,k) for k,v in label_names.items())

true_labels = [label_names[np.argmax(label)] for label in labels]

predicted_labels = [label_names[np.argmax(pred)] for pred in predictions]

#  Displaying the 10 images with their true and predicted labels
import matplotlib.pyplot as plt

fig, axs = plt.subplots(2, 5, figsize=(15, 6))  
for i, ax in enumerate(axs.flat):
    ax.imshow(images[i]) #images[i] 
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
