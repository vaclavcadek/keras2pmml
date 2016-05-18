from xml.dom.minidom import Document

from keras.models import Sequential
from sklearn.preprocessing import StandardScaler

SUPPORTED_MODELS = frozenset([Sequential])
SUPPORTED_TRANSFORMERS = frozenset([StandardScaler])
SUPPORTED_ACTIVATIONS = {
    'tanh': 'tanh',
    'sigmoid': 'logistic'
}


def _validate_inputs(model, transformer, feature_names, target_values):
    print('[X] Performing model validation.')
    if not type(model) in SUPPORTED_MODELS:
        raise TypeError("Provided model is not supported.")
    if not model.built:
        raise TypeError("Provide a fitted model.")
    if transformer is not None and not type(transformer) in SUPPORTED_TRANSFORMERS:
        raise TypeError("Provided transformer is not supported.")
    if model.input_shape[1] != len(feature_names):
        print('[!] Network input shape does not match provided feature names - using generic names instead.')
        feature_names = ['x{}'.format(i) for i in range(model.input_shape[1])]
    if model.output_shape[1] != len(target_values):
        print('[!] Network output shape does not match provided target values - using generic names instead.')
        target_values = ['y{}'.format(i) for i in range(model.output_shape[1])]
    print('[X] Model validation successful.')
    return feature_names, target_values


def _generate_metadata(doc, root, kwargs):
    description = kwargs.get('description', None)
    header = doc.createElement('Header')
    root.appendChild(header)
    header.attributes['copyright'] = kwargs.get('copyright', '')
    header.attributes['description'] = description
    timestamp = doc.createElement('Timestamp')
    header.appendChild(timestamp)


def _generate_variables(doc, root, neural_network, transformer, feature_names, target_name, target_values):
    data_dict = doc.createElement('DataDictionary')
    root.appendChild(data_dict)
    data_field = doc.createElement('DataField')
    data_dict.appendChild(data_field)
    data_field.attributes['name'] = target_name
    data_field.attributes['dataType'] = 'string'
    data_field.attributes['optype'] = 'categorical'
    mining_schema = doc.createElement('MiningSchema')
    neural_network.appendChild(mining_schema)
    mining_field = doc.createElement('MiningField')
    mining_schema.appendChild(mining_field)
    mining_field.attributes['name'] = target_name
    mining_field.attributes['usageType'] = 'target'
    output = doc.createElement('Output')
    neural_network.appendChild(output)
    neural_inputs = doc.createElement('NeuralInputs')
    neural_network.appendChild(neural_inputs)
    for i, f in enumerate(feature_names):
        data_field = doc.createElement('DataField')
        data_dict.appendChild(data_field)
        data_field.attributes['name'] = f
        data_field.attributes['dataType'] = 'double'
        data_field.attributes['optype'] = 'continuous'

        mining_field = doc.createElement('MiningField')
        mining_schema.appendChild(mining_field)
        mining_field.attributes['name'] = f

        neural_input = doc.createElement('NeuralInput')
        neural_inputs.appendChild(neural_input)
        neural_input.attributes['id'] = '0,{}'.format(i)
        derived_field = doc.createElement('DerivedField')
        neural_input.appendChild(derived_field)
        derived_field.attributes['optype'] = 'continuous'
        derived_field.attributes['dataType'] = 'double'
        if transformer is not None:
            norm_continuous = doc.createElement('NormContinuous')
            derived_field.appendChild(norm_continuous)
            norm_continuous.attributes['field'] = f
            ln1 = doc.createElement('LinearNorm')
            norm_continuous.appendChild(ln1)
            ln2 = doc.createElement('LinearNorm')
            norm_continuous.appendChild(ln2)
            ln1.attributes['orig'] = '0.0'
            ln1.attributes['norm'] = str(- transformer.mean_[i] / transformer.scale_[i])
            ln2.attributes['orig'] = str(transformer.mean_[i])
            ln2.attributes['norm'] = '0.0'
        else:
            field_ref = doc.createElement('FieldRef')
            derived_field.appendChild(field_ref)
            field_ref.attributes['field'] = f
    for t in target_values:
        value_field = doc.createElement('Value')
        data_field.appendChild(value_field)
        value_field.attributes['value'] = t

        output_field = doc.createElement('OutputField')
        output.appendChild(output_field)
        output_field.attributes['name'] = 'probability_{}'.format(t)
        output_field.attributes['feature'] = 'probability'
        output_field.attributes['value'] = t


def _generate_layers(doc, estimator, neural_network, target_name, target_values):
    layer_weights = estimator.get_weights()[0::2]
    layer_biases = estimator.get_weights()[1::2]
    layer_activations = map(lambda x: x['config']['activation'], estimator.get_config())
    last_layer = len(layer_weights) - 1
    for layer, params in enumerate(zip(layer_weights, layer_biases, layer_activations)):
        weights = params[0]
        biases = params[1]
        activation = params[2]
        neural_layer = doc.createElement('NeuralLayer')
        neural_network.appendChild(neural_layer)
        neural_layer.attributes['activationFunction'] = SUPPORTED_ACTIVATIONS[activation]
        rows = weights.shape[0]
        cols = weights.shape[1]
        if layer == last_layer:
            output_layer = doc.createElement('NeuralOutputs')
            neural_network.appendChild(output_layer)
        for j in range(cols):
            neuron = doc.createElement('Neuron')
            neural_layer.appendChild(neuron)
            neuron.attributes['id'] = '{},{}'.format(layer + 1, j)
            neuron.attributes['bias'] = str(biases[j])
            if layer == last_layer:
                neural_output = doc.createElement('NeuralOutput')
                output_layer.appendChild(neural_output)
                neural_output.attributes['outputNeuron'] = '{},{}'.format(layer + 1, j)
                derived_field = doc.createElement('DerivedField')
                neural_output.appendChild(derived_field)
                derived_field.attributes['optype'] = 'continuous'
                derived_field.attributes['dataType'] = 'double'
                norm_discrete = doc.createElement('NormDiscrete')
                derived_field.appendChild(norm_discrete)
                norm_discrete.attributes['field'] = target_name
                norm_discrete.attributes['value'] = target_values[j]
            for i in range(rows):
                connection = doc.createElement('Con')
                connection.attributes['from'] = '{},{}'.format(layer, i)
                connection.attributes['weight'] = str(weights[i, j])
                neuron.appendChild(connection)


def keras2pmml(estimator, transformer, file, **kwargs):
    feature_names = kwargs.get('feature_names', [])
    target_name = kwargs.get('target_name', 'class')
    target_values = kwargs.get('target_values', [])
    model_name = kwargs.get('model_name', None)

    feature_names, target_values = _validate_inputs(estimator, transformer, feature_names, target_values)

    doc = Document()
    root = doc.createElement('PMML')
    root.attributes['xmlns'] = 'http://www.dmg.org/PMML-4_2'
    root.attributes['version'] = '4.2.1'
    doc.appendChild(root)
    _generate_metadata(doc, root, kwargs)

    neural_network = doc.createElement('NeuralNetwork')
    _generate_variables(doc, root, neural_network, transformer, feature_names, target_name, target_values)
    root.appendChild(neural_network)
    neural_network.attributes['modelName'] = model_name
    neural_network.attributes['functionName'] = 'classification'

    _generate_layers(doc, estimator, neural_network, target_name, target_values)

    doc.writexml(open(file, 'w'), indent=' ', addindent=' ', newl='\n')
