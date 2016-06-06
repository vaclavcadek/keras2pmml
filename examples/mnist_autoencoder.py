from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense
from keras2pmml import keras2pmml
from sklearn.preprocessing import MinMaxScaler

(X_train, _), (X_test, _) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 28 * 28).astype(float)
X_test = X_test.reshape(X_test.shape[0], 28 * 28).astype(float)

mms = MinMaxScaler()
X_train_scaled = mms.fit_transform(X_train)
X_test_scaled = mms.transform(X_test)

autoencoder = Sequential()
# encoding ~ input layer
autoencoder.add(Dense(input_dim=X_train.shape[1], output_dim=100, activation='tanh'))

# compression layer
autoencoder.add(Dense(input_dim=100, output_dim=100, activation='tanh'))

# decoding ~ output layer
autoencoder.add(Dense(input_dim=100, output_dim=X_train.shape[1], activation='sigmoid'))

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder.fit(X_train, X_train, nb_epoch=10, batch_size=256, verbose=3, validation_data=(X_test, X_test))

# export same way as Iris without feature names or target value names - it will then use general x0...x783 and y0..y783
params = {
    'copyright': 'Václav Čadek',
    'description': 'Simple Keras model for Autoencoding of MNIST dataset.',
    'model_name': 'MNIST Autoencoder'
}

keras2pmml(autoencoder, mms, 'mnist_autoencoder.pmml', **params)