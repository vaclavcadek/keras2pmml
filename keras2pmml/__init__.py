try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
from datetime import datetime
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler, MinMaxScaler

SUPPORTED_MODELS = frozenset([Sequential])
SUPPORTED_TRANSFORMERS = frozenset([StandardScaler, MinMaxScaler])
SUPPORTED_ACTIVATIONS = {
    'tanh': 'tanh',
    'sigmoid': 'logistic'
}


def _validate_inputs(model, transformer, feature_names, target_values):
    print('[x] Performing model validation.')
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
    print('[x] Model validation successful.')
    return feature_names, target_values


def _generate_header(root, kwargs):
    description = kwargs.get('description', None)
    copyright = kwargs.get('copyright', None)
    header = ET.SubElement(root, 'Header')
    if copyright:
        header.set('copyright', copyright)
    if description:
        header.set('description', description)
    timestamp = ET.SubElement(header, 'Timestamp')
    timestamp.text = str(datetime.now())
    return header


def _generate_data_dictionary(root, feature_names, target_name, target_values):
    data_dict = ET.SubElement(root, 'DataDictionary')
    data_field = ET.SubElement(data_dict, 'DataField')
    data_field.set('name', target_name)
    data_field.set('dataType', 'string')
    data_field.set('optype', 'categorical')
    print('[x] Generating Data Dictionary:')
    for t in target_values:
        value = ET.SubElement(data_field, 'Value')
        value.set('value', t)
    for f in feature_names:
        data_field = ET.SubElement(data_dict, 'DataField')
        data_field.set('name', f)
        data_field.set('dataType', 'double')
        data_field.set('optype', 'continuous')
        print('\t[-] {}...OK!'.format(f))
    return data_dict


def _generate_neural_network(root, estimator, transformer, feature_names, target_name, target_values, model_name=None):
    neural_network = ET.SubElement(root, 'NeuralNetwork')
    neural_network.set('functionName', 'classification')
    neural_network.set('activationFunction', 'logistic')
    if model_name:
        neural_network.set('modelName', model_name)
    _generate_mining_schema(neural_network, feature_names, target_name)
    _generate_output(neural_network, target_values)
    _generate_neural_inputs(neural_network, transformer, feature_names)
    _generate_neural_layers(neural_network, estimator)
    _generate_neural_outputs(neural_network, estimator, target_name, target_values)
    return neural_network


def _generate_mining_schema(neural_network, feature_names, target_name):
    mining_schema = ET.SubElement(neural_network, 'MiningSchema')
    mining_field = ET.SubElement(mining_schema, 'MiningField')
    mining_field.set('name', target_name)
    mining_field.set('usageType', 'target')
    for f in feature_names:
        mining_field = ET.SubElement(mining_schema, 'MiningField')
        mining_field.set('name', f)
        mining_field.set('usageType', 'active')
    return mining_schema


def _generate_output(neural_network, target_values):
    output = ET.SubElement(neural_network, 'Output')
    for t in target_values:
        output_field = ET.SubElement(output, 'OutputField')
        output_field.set('name', 'probability_{}'.format(t))
        output_field.set('feature', 'probability')
        output_field.set('value', t)
    return output


def _generate_neural_inputs(neural_network, transformer, feature_names):
    neural_inputs = ET.SubElement(neural_network, 'NeuralInputs')
    for i, f in enumerate(feature_names):
        neural_input = ET.SubElement(neural_inputs, 'NeuralInput')
        neural_input.set('id', '0,{}'.format(i))
        derived_field = ET.SubElement(neural_input, 'DerivedField')
        derived_field.set('optype', 'continuous')
        derived_field.set('dataType', 'double')
        if transformer is not None and type(transformer) in SUPPORTED_TRANSFORMERS:
            if isinstance(transformer, StandardScaler):
                if transformer.mean_[i] == 0:
                    norm_discrete = ET.SubElement(derived_field, 'NormDiscrete')
                    norm_discrete.set('field', f)
                    norm_discrete.set('value', '0.0')
                    print(
                        '[!] {field} has zero mean, avoiding scaling. Check whether your data does not contains only one value!'.format(
                            field=f))
                else:
                    norm_continuous = ET.SubElement(derived_field, 'NormContinuous')
                    norm_continuous.set('field', f)
                    ln1 = ET.SubElement(norm_continuous, 'LinearNorm')
                    ln2 = ET.SubElement(norm_continuous, 'LinearNorm')
                    ln1.set('orig', '0.0')
                    ln1.set('norm', str(- transformer.mean_[i] / transformer.scale_[i]))
                    ln2.set('orig', str(transformer.mean_[i]))
                    ln2.set('norm', '0.0')
            elif isinstance(transformer, MinMaxScaler):
                norm_continuous = ET.SubElement(derived_field, 'NormContinuous')
                norm_continuous.set('field', f)
                ln1 = ET.SubElement(norm_continuous, 'LinearNorm')
                ln2 = ET.SubElement(norm_continuous, 'LinearNorm')
                ln1.set('orig', '0.0')
                ln1.set('norm', str(- transformer.data_min_[i] / (transformer.data_max_[i] - transformer.data_min_[i])))
                ln2.set('orig', str(transformer.data_min_[i]))
                ln2.set('norm', '0.0')
        else:
            norm_discrete = ET.SubElement(derived_field, 'FieldRef')
            norm_discrete.set('field', f)


def _generate_neural_layers(neural_network, estimator):
    layer_weights = estimator.get_weights()[0::2]
    layer_biases = estimator.get_weights()[1::2]
    layer_activations = map(lambda x: x['config']['activation'], estimator.get_config())
    for layer, params in enumerate(zip(layer_weights, layer_biases, layer_activations)):
        weights = params[0].astype(str)
        biases = params[1].astype(str)
        activation = params[2]
        neural_layer = ET.SubElement(neural_network, 'NeuralLayer')
        neural_layer.set('activationFunction', SUPPORTED_ACTIVATIONS[activation])
        rows = weights.shape[0]
        cols = weights.shape[1]
        for j in range(cols):
            neuron = ET.SubElement(neural_layer, 'Neuron')
            neuron.set('id', '{},{}'.format(layer + 1, j))
            neuron.set('bias', str(biases[j]))
            for i in range(rows):
                connection = ET.SubElement(neuron, 'Con')
                connection.set('from', '{},{}'.format(layer, i))
                connection.set('weight', weights[i, j])


def _generate_neural_outputs(neural_network, estimator, target_name, target_values):
    num_layers = len(estimator.layers)
    neural_outputs = ET.SubElement(neural_network, 'NeuralOutputs')
    for i in range(len(target_values)):
        neural_output = ET.SubElement(neural_outputs, 'NeuralOutput')
        neural_output.set('outputNeuron', '{},{}'.format(num_layers, i))
        derived_field = ET.SubElement(neural_output, 'DerivedField')
        derived_field.set('optype', 'continuous')
        derived_field.set('dataType', 'double')
        norm_discrete = ET.SubElement(derived_field, 'NormDiscrete')
        norm_discrete.set('field', target_name)
        norm_discrete.set('value', target_values[i])


def keras2pmml(estimator, transformer=None, file=None, **kwargs):
    """
    Exports Keras model as PMML.

    :param estimator: Keras model to be exported as PMML (for supported models - see bellow).
    :param transformer: if provided then scaling is applied to data fields.
    :param file: name of the file where the PMML will be exported.
    :param kwargs: set of params that affects PMML metadata - see documentation for details.
    :return: XML element tree
    """

    feature_names = kwargs.get('feature_names', [])
    target_name = kwargs.get('target_name', 'class')
    target_values = kwargs.get('target_values', [])
    model_name = kwargs.get('model_name', None)

    feature_names, target_values = _validate_inputs(estimator, transformer, feature_names, target_values)

    pmml = ET.Element('PMML')
    pmml.set('version', '4.2.1')
    pmml.set('xmlns', 'http://www.dmg.org/PMML-4_2')
    _generate_header(pmml, kwargs)
    _generate_data_dictionary(pmml, feature_names, target_name, target_values)
    _generate_neural_network(pmml, estimator, transformer, feature_names, target_name, target_values, model_name)

    tree = ET.ElementTree(pmml)
    print('[x] Generation of PMML successful.')
    if file:
        tree.write(file, encoding='utf-8', xml_declaration=True)
    return tree
