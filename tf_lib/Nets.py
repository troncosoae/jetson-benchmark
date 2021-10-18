from tensorflow.keras.models import Sequential
from tensorflow.keras import layers


MediumNet = Sequential([
        layers.Conv2D(
            filters=6, kernel_size=5, padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2), strides=2),
        layers.Conv2D(
            filters=16, kernel_size=5, padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2), strides=2),
        layers.Flatten(),
        layers.Dense(120, activation='relu'),
        layers.Dense(84, activation='relu'),
        layers.Dense(10)
    ])
