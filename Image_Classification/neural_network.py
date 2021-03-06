import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

if __name__ == "__main__":
    data = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = data.load_data()
    class_names = ['T-shirt/Top', 'Trousers', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    train_images = train_images/255.0
    test_images = test_images/255.0

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(10, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_images, train_labels, epochs=10)
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print("Tested Acc", test_acc)
    prediction = model.predict(test_images)
    print(class_names[np.argmax(prediction[0])])
    plt.imshow(test_images[0], cmap=plt.cm.binary)
    plt.show()
    pass
