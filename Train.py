import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def load_and_normalize_data():
    # Load MNIST dataset
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Normalize the data to [0, 1]
    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)
    
    # Single example image and label
    single_image = x_train[0]
    single_label = y_train[0]
    
    return (x_train, y_train), (x_test, y_test), (single_image, single_label)


def build_and_compile_model():
    # Sequential model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    # Compile the model
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

def train_and_evaluate_model(model, x_train, y_train, x_test, y_test):
    # Train the model
    history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
    
    # Evaluate the model
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {accuracy * 100:.2f}%")
    return history

def plot_metrics(history):
    # Plot accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

def plot_single_image(image, label):
    # Plot single image
    plt.imshow(image, cmap='gray')
    plt.title(f'Label: {label}')
    plt.colorbar()
    plt.show()

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test), (single_image, single_label) = load_and_normalize_data()
    model = build_and_compile_model()
    history = train_and_evaluate_model(model, x_train, y_train, x_test, y_test)
    plot_metrics(history)
    model.save('model/handwritten.keras')  # Save the model once after training

    # plot_single_image(single_image, single_label)
    # Adjust numpy print options
    # np.set_printoptions(linewidth=np.inf)
    # print(single_image)




    