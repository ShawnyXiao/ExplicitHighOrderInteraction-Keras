import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

from xdeepfm import xDeepFM

if __name__ == '__main__':
    x = np.random.random((1000, 50, 128))
    y = np.random.randint(0, 2, (1000, 1))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2020)

    print('Build model...')
    xdeepfm = xDeepFM()
    xdeepfm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    print('Train...')
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, mode='max')
    xdeepfm.fit(x_train, y_train,
                batch_size=32,
                epochs=10,
                callbacks=[early_stopping],
                validation_data=(x_test, y_test))

    print('Test...')
    result = xdeepfm.predict(x_test)
