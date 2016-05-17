from xml.dom.minidom import Document

from keras.models import Sequential
from sklearn.preprocessing import StandardScaler


def keras2pmml(estimator, scaler, file, **kwargs):
    if not isinstance(estimator, Sequential):
        raise TypeError("The estimator object is not an instance of " + Sequential.__name__)
    if scaler is not None and not isinstance(scaler, StandardScaler):
        raise TypeError("The estimator object is not an instance of " + StandardScaler.__name__)
    target = kwargs.get('target', 'class')
    target_values = kwargs.get('target_values', None)
    feature_names = kwargs.get('feature_names', None)
    copyright = kwargs.get('copyright', None)
    description = kwargs.get('description', '')
    model_name = kwargs.get('model_name', None)

    doc = Document()

    root = doc.createElement('PMML')
    root.attributes['xmlns'] = 'http://www.dmg.org/PMML-4_1'
    root.attributes['version'] = '4.1'
    doc.appendChild(root)

    header = doc.createElement('Header')
    header.attributes['copyright'] = copyright
    header.attributes['description'] = description
    timestamp = doc.createElement('Timestamp')
    root.appendChild(header)
    header.appendChild(timestamp)

    data_dict = doc.createElement('DataDictionary')
    root.appendChild(data_dict)
    for f in feature_names:
        data_field = doc.createElement('DataField')
        data_field.attributes['name'] = f
        data_field.attributes['dataType'] = 'double'
        data_field.attributes['optype'] = 'continuous'
        data_dict.appendChild(data_field)

    data_field = doc.createElement('DataField')
    data_field.attributes['name'] = target
    data_field.attributes['dataType'] = 'string'
    data_field.attributes['optype'] = 'categorical'
    data_dict.appendChild(data_field)
    for t in target_values:
        value_field = doc.createElement('Value')
        value_field.attributes['value'] = t
        data_field.appendChild(value_field)

    neural_network = doc.createElement('NeuralNetwork')
    neural_network.attributes['modelName'] = model_name
    neural_network.attributes['functionName'] = 'classification'
    root.appendChild(neural_network)

    mining_schema = doc.createElement('MiningSchema')
    neural_network.appendChild(mining_schema)
    mining_field = doc.createElement('MiningField')
    mining_field.attributes['name'] = target
    mining_field.attributes['usageType'] = 'target'
    mining_schema.appendChild(mining_field)
    for f in feature_names:
        mining_field = doc.createElement('MiningField')
        mining_field.attributes['name'] = f
        mining_schema.appendChild(mining_field)

    output = doc.createElement('Output')
    neural_network.appendChild(output)
    for t in target_values:
        output_field = doc.createElement('OutputField')
        output_field.attributes['name'] = 'probability_{}'.format(t)
        output_field.attributes['feature'] = 'probability'
        output_field.attributes['value'] = t
        output.appendChild(output_field)

    neural_inputs = doc.createElement('NeuralInputs')
    for i, f in enumerate(feature_names):
        neural_input = doc.createElement('NeuralInput')
        neural_input.attributes['id'] = '0,{}'.format(i)
        neural_inputs.appendChild(neural_input)
        derived_field = doc.createElement('DerivedField')
        derived_field.attributes['optype'] = 'continuous'
        derived_field.attributes['dataType'] = 'double'
        neural_input.appendChild(derived_field)
        norm_continuous = doc.createElement('NormContinuous')
        norm_continuous.attributes['field'] = f
        derived_field.appendChild(norm_continuous)
        ln1 = doc.createElement('LinearNorm')
        ln2 = doc.createElement('LinearNorm')
        ln1.attributes['orig'] = '0.0'
        ln1.attributes['norm'] = str(- scaler.mean_[i] / scaler.scale_[i])
        ln2.attributes['orig'] = str(scaler.mean_[i])
        ln2.attributes['norm'] = '0.0'
        norm_continuous.appendChild(ln1)
        norm_continuous.appendChild(ln2)
    neural_network.appendChild(neural_inputs)

    neural_weights = estimator.get_weights()[0::2]
    neural_biases = estimator.get_weights()[1::2]
    neural_activations = ['tanh', 'tanh', 'logistic']
    last_layer = len(neural_weights) - 1
    for layer, params in enumerate(zip(neural_weights, neural_biases, neural_activations)):
        weights = params[0]
        biases = params[1]
        activation = params[2]
        neural_layer = doc.createElement('NeuralLayer')
        neural_layer.attributes['activationFunction'] = activation
        neural_network.appendChild(neural_layer)
        rows = weights.shape[0]
        cols = weights.shape[1]
        if layer == last_layer:
            output_layer = doc.createElement('NeuralOutputs')
            neural_network.appendChild(output_layer)
        for j in range(cols):
            neuron = doc.createElement('Neuron')
            neuron.attributes['id'] = '{},{}'.format(layer + 1, j)
            neuron.attributes['bias'] = str(biases[j])
            neural_layer.appendChild(neuron)
            if layer == last_layer:
                neural_output = doc.createElement('NeuralOutput')
                neural_output.attributes['outputNeuron'] = '{},{}'.format(layer + 1, j)
                output_layer.appendChild(neural_output)
                derived_field = doc.createElement('DerivedField')
                derived_field.attributes['optype'] = 'continuous'
                derived_field.attributes['dataType'] = 'double'
                norm_discrete = doc.createElement('NormDiscrete')
                norm_discrete.attributes['field'] = target
                norm_discrete.attributes['value'] = target_values[j]
                derived_field.appendChild(norm_discrete)
                neural_output.appendChild(derived_field)
            for i in range(rows):
                connection = doc.createElement('Con')
                connection.attributes['from'] = '{},{}'.format(layer, i)
                connection.attributes['weight'] = str(weights[i, j])
                neuron.appendChild(connection)

    doc.writexml(open(file, 'w'), indent=' ', addindent=' ', newl='\n')
