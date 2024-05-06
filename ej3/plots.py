
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def plot_fit(history, epochs):
    # Grafica la precisión y pérdida de entrenamiento y validación
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

def preprocess_image(image_path, target_size=(150, 150)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    return img

def plot_predictions(model, img_pred_path, class_names, sx=6, sy=3, figsize=(50, 50)):
    test_image_paths = list(img_pred_path.glob("*.jpg"))

    fig, axs = plt.subplots(sx, sy, figsize=figsize)

    i = 0
    for x in range(0, sx):
        for y in range(0,sy):

            image_path = test_image_paths[i]
            image = preprocess_image(image_path)
            image_show = preprocess_image(image_path,(512,512))
            axs[x][y].imshow(image_show)
            image = np.expand_dims(image, axis=0)  # Add batch dimension

            # Perform prediction
            prediction = model.predict(image)
            print(prediction)
            predicted_class_index = np.argmax(prediction)
            predicted_class = class_names[predicted_class_index]        

            axs[x][y].set_title(f"Predicted: {predicted_class}")
            i+=1

    plt.tight_layout()
    plt.show()