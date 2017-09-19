import numpy as np
from TripleTriad.feature import *
from keras.models import Sequential, Model, model_from_json
from keras.layers import convolutional, merge, Input, BatchNormalization, Dense
from keras.layers.core import Activation, Flatten
from keras.engine.topology import Layer
from keras import backend as K
import json

DEFAULT_FEATURES = [
    "board_numbers",
    "opp_handcards",
    "self_handcards",
    "turn"
    ]

DEFAULT_NN_PARAMETERS = {
    "layers": 3,
    "board_size": 3,
    "activation": "relu",
    "output_activation": "softmax"
    }

DEFAULT_MODEL_OUTPUT_PATH = "test_nn_model.json"

class RandomPolicy():
    # This is an equiprobable policy that simply randomly pick one move from all the legal moves
        
    def get_action(self, state):
        moves = state.get_legal_moves()
        return np.random.choice(moves)
    
class NNPolicy():
    
    def __init__(self, params = DEFAULT_NN_PARAMETERS, features = DEFAULT_FEATURES, model_save_path = DEFAULT_MODEL_OUTPUT_PATH, 
                 model_load_path = None):
        self.params = params 
        self.features = features
        self.model_save_path = model_save_path
        if model_load_path is not None:
            self.model = self.load_model(model_load_path)
        else:
            self.model = self.create_neural_network()
    
    def create_neural_network(self):
        network = Sequential()
        
        # Draft Network Architecture for testing purpose
        # TODO - replace with a real architecture
        network.add(Dense(
            input_shape=(get_feature_dim(self.features), self.params["board_size"], self.params["board_size"]),
            activation=self.params["activation"]))
        for _ in range(self.params["layers"]):
            network.add(Dense(activation=self.params["activation"]))
        model.add(Flatten())
        network.add(Bias())
        network.add(Activation(self.params["output_activation"]))
        
    def get_action(self, state):
        moves = state.get_legal_moves()
        if len(moves) == 0:
            return None
        # TODO pick the right action from neural network forward output
            
    def forward(self, state):   
        forward_function = K.function([self.model.input, K.learning_phase()], [self.model.output])
        return forward_function(state2feature(state, self.features))
    
    def fit(self, states, actions, rewards):
        # the fit method will update the policy by a batch of simulated experiences
        # states: n x dim array. n is the number of samples to be trained. dim is the dimenstion of all the features
        # actions: n x (board_size * board_size) array. A one-hot array for each possible action tuple (x, y)
        # rewards: n x 1 array. With either +1/0/-1 for win/tie/lose
        self.model.optimizer.lr = K.abs(optimizer.lr) * (+1 if won else -1)
        self.model.train_on_batch(states, actions)
        
    def save_model(self, weights_file=None):
        """write the network model and preprocessing features to the specified file

        If a weights_file (.hdf5 extension) is also specified, model weights are also
        saved to that file and will be reloaded automatically in a call to load_model
        """
        object_specs = {
            'class': self.__class__.__name__,
            'keras_model': self.model.to_json(),
            'feature_list': self.features
        }
        if weights_file is not None:
            self.model.save_weights(weights_file)
            object_specs['weights_file'] = weights_file
        # use the json module to write object_specs to file
        with open(self.model_save_path, 'w') as f:
            json.dump(object_specs, f)
    
    def load_model(self, model_load_path):
        """
        Load the neural network model from a json file in the self.model_load_path
        """
        with open(model_load_path, 'r') as f:
            object_specs = json.load(f)
        model = model_from_json(object_specs['keras_model'], custom_objects={'Bias': Bias})
        if 'weights_file' in object_specs:
            model.load_weights(object_specs['weights_file'])
        return model
        
class Bias(Layer):
    """Custom keras layer that simply adds a scalar bias to each location in the input

    Largely copied from the keras docs:
    http://keras.io/layers/writing-your-own-keras-layers/#writing-your-own-keras-layers
    """

    def __init__(self, **kwargs):
        super(Bias, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = K.zeros(input_shape[1:])
        self.trainable_weights = [self.W]

    def call(self, x, mask=None):
        return x + self.W