from keras2pmml import keras2pmml
from sklearn.datasets import load_iris
import numpy as np
import theano
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense

iris = load_iris()
X = iris.data
y = iris.target

theano.config.floatX = 'float32'
X = X.astype(theano.config.floatX)
y = y.astype(np.int32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, stratify=y)

std = StandardScaler()
X_train_scaled = std.fit_transform(X_train)
X_test_scaled = std.transform(X_test)
y_train_ohe = np_utils.to_categorical(y_train)
y_test_ohe = np_utils.to_categorical(y_test)

model = Sequential()
model.add(Dense(input_dim=X_train.shape[1], output_dim=5, activation='tanh'))
model.add(Dense(input_dim=5, output_dim=y_test_ohe.shape[1], activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='sgd')
model.fit(X_train_scaled, y_train_ohe, nb_epoch=10, batch_size=1, verbose=3,
          validation_data=(X_test_scaled, y_test_ohe))

params = {
    'feature_names': ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
    'target_values': ['setosa', 'virginica', 'versicolor'],
    'target_name': 'specie',
    'copyright': 'Václav Čadek',
    'description': 'Simple Keras model for Iris dataset.',
    'model_name': 'Iris Model'
}

keras2pmml(model, None, 'keras_iris.pmml', **params)
