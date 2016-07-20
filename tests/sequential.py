import unittest

from keras2pmml import keras2pmml
from sklearn.datasets import load_iris
import numpy as np
import theano
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense


class GenericFieldsTestCase(unittest.TestCase):
    def setUp(self):
        iris = load_iris()

        theano.config.floatX = 'float32'
        X = iris.data.astype(theano.config.floatX)
        y = iris.target.astype(np.int32)
        y_ohe = np_utils.to_categorical(y)

        model = Sequential()
        model.add(Dense(input_dim=X.shape[1], output_dim=5, activation='tanh'))
        model.add(Dense(input_dim=5, output_dim=y_ohe.shape[1], activation='sigmoid'))
        model.compile(loss='categorical_crossentropy', optimizer='sgd')
        model.fit(X, y_ohe, nb_epoch=10, batch_size=1, verbose=3, validation_data=None)

        params = {'copyright': 'Václav Čadek', 'model_name': 'Iris Model'}
        self.model = model
        self.pmml = keras2pmml(self.model, **params)
        self.num_inputs = self.model.input_shape[1]
        self.num_outputs = self.model.output_shape[1]
        self.num_connection_layers = len(self.model.layers)
        self.features = ['x{}'.format(i) for i in range(self.num_inputs)]
        self.class_values = ['y{}'.format(i) for i in range(self.num_outputs)]

    def test_data_dict(self):
        continuous_fields = self.pmml.findall("DataDictionary/DataField/[@optype='continuous']")
        categorical_field = self.pmml.findall("DataDictionary/DataField/[@optype='categorical']")
        self.assertEquals(len(continuous_fields), self.num_inputs, 'Correct number of continuous fields.')
        self.assertEquals(len(categorical_field), 1, 'Exactly one categorical field in data dictionary.')
        categorical_name = categorical_field[0].attrib.get('name', None)
        self.assertEquals(categorical_name, 'class', 'Correct target variable name.')
        output_values = categorical_field[0].findall('Value')
        self.assertEqual(len(output_values), self.num_outputs, 'Correct number of output values.')
        self.assertListEqual(
            [ov.attrib['value'] for ov in output_values],
            self.class_values
        )
        self.assertListEqual(
            [ov.attrib['name'] for ov in continuous_fields],
            self.features
        )

    def test_mining_schema(self):
        target_field = self.pmml.findall("NeuralNetwork/MiningSchema/MiningField/[@usageType='target']")
        active_fields = self.pmml.findall("NeuralNetwork/MiningSchema/MiningField/[@usageType='active']")
        self.assertEquals(len(active_fields), self.num_inputs, 'Correct number of active fields.')
        self.assertEquals(len(target_field), 1, 'Exactly one target field in mining schema.')
        target_name = target_field[0].attrib.get('name', None)
        self.assertEquals(target_name, 'class', 'Correct target field name.')
        self.assertListEqual(
            [ov.attrib['name'] for ov in active_fields],
            self.features
        )

    def test_input(self):
        field_refs = self.pmml.findall("NeuralNetwork/NeuralInputs/NeuralInput/DerivedField/FieldRef")
        for fr in field_refs:
            if fr.attrib.get('field') not in self.features:
                self.fail('Field mapped to non-existing field.')

    def test_output(self):
        output_fields = self.pmml.findall("NeuralNetwork/Output/OutputField/[@feature='probability']")
        self.assertEqual(len(output_fields), self.num_outputs, 'Correct number of output fields.')
        self.assertListEqual(
            [of.attrib['name'] for of in output_fields],
            ['probability_{}'.format(v) for v in self.class_values]
        )

    def test_topology(self):
        neural_inputs = self.pmml.findall('NeuralNetwork/NeuralInputs/NeuralInput')
        neural_outputs = self.pmml.findall('NeuralNetwork/NeuralOutputs/NeuralOutput')
        neural_layers = self.pmml.findall('NeuralNetwork/NeuralLayer')
        self.assertEqual(len(neural_inputs), self.num_inputs, 'Correct number of input neurons.')
        self.assertEqual(len(neural_outputs), self.num_outputs, 'Correct number of output neurons.')
        self.assertEquals(len(neural_layers), self.num_connection_layers, 'Correct number of layers.')
        for i, l in enumerate(neural_layers):
            weights = self.model.layers[i].get_weights()[0]
            biases = self.model.layers[i].get_weights()[1]
            for j, n in enumerate(l.findall('Neuron')):
                self.assertListEqual(
                    [float(c.attrib['weight']) for c in n.findall('Con')], weights[:, j].tolist(),
                    'Verify correct weights and that is fully-connected from previous layer.'
                )
                self.assertEquals(n.attrib['bias'], biases[j])
