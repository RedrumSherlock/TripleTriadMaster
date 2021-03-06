import TripleTriad.feature as fe
import TripleTriad.game as gm
from TripleTriad.player.policy import Policy
import TripleTriad.game_helper as Helper

import numpy as np
import random
import copy
import json

from keras.models import Sequential, Model, model_from_json, clone_model
from keras.layers import Input, BatchNormalization, Dense, MaxPooling1D, SeparableConv1D
from keras.layers.core import Activation, Flatten, Dropout
from keras.engine.topology import Layer
from keras import backend as K


DEFAULT_NN_PARAMETERS = {
    "conv_layers": 9,
    "filters": 128,
    "dense_layers": 2,
    "units": 256,
    "activation": "relu",
    "output_activation": "softmax",
    "dropout": 0
    }

DEFAULT_MODEL_OUTPUT_PATH = "model.json"

class NNPolicy(Policy):
    
    def __init__(self, params = DEFAULT_NN_PARAMETERS, features = fe.DEFAULT_FEATURES, model_save_path = DEFAULT_MODEL_OUTPUT_PATH, 
                 model_load_path = None):
        self.params = params 
        self.features = features
        self.model_save_path = model_save_path
        if model_load_path is not None:
            self.model = self.load_model(model_load_path)
        else:
            self.model = self.create_neural_network()
        self.predict_func = K.function([self.model.input, K.learning_phase()], self.model.output)
    
    def create_neural_network(self):
        
        """
        Draft Network Architecture for testing purpose
        Input: 16x10 size
        
        Try with 3 hidden layers for now. Each layer has 120 units with a dropout rate at 0.25 and relu as the activation layer
        Use Flatten to map to a one dimension output with size 19 (first 10 maps to the cards to pick, and the last 9 map to the board 
        to drop the card)
        
        There should be two output layers: one for the card and one for the move
        """
        
        state_input = Input(shape = (fe.get_feature_dim(self.features), 2 * gm.START_HANDS))
        
        # Testing for Conv1d layer
        
        x = SeparableConv1D(self.params["filters"], 5, padding='same', data_format='channels_first', name="Conv_kernal_5")(state_input)
        x = BatchNormalization()(x)
        x = Activation(self.params["activation"])(x)
        x = MaxPooling1D(padding='same')(x)
        
        for i in range(self.params["conv_layers"]):
            x = SeparableConv1D(self.params["filters"], 3, padding='same', data_format='channels_first', name="Conv1D_{}".format(i+1))(x)
            x = BatchNormalization()(x)
            x = Activation(self.params["activation"])(x)
            x = MaxPooling1D(padding='same')(x)
        
        x = Flatten()(x)
            
        for i in range(self.params["dense_layers"]):
            x = Dense(self.params["units"], activation=self.params["activation"], name="Dense_{}".format(i+1))(x)
            if self.params["dropout"] > 0:
                x = Dropout(self.params["dropout"])(x)
                
        
        x1 = Bias()(x)
        x2 = Bias()(x)
        card_output = Dense(2 * gm.START_HANDS, activation=self.params["output_activation"], name='card_output')(x1)
        move_output = Dense(gm.BOARD_SIZE ** 2, activation=self.params["output_activation"], name='move_output')(x2)
        network = Model(input=state_input, output=[card_output, move_output])
        return network
    
    def print_network(self):
        self.model.summary()
            
    def clone(self):
        new_policy = NNPolicy()
        new_policy.params = self.params.copy()
        new_policy.features = copy.copy(self.features)
        new_policy.model_save_path = self.model_save_path
        new_policy.model = clone_model(self.model)
        return new_policy
        
    def get_action(self, state, card_index = -1, board_index = -1):
        # This will return the (card, move) pair, where card is the Card object, and move is the (x, y) tuple
        if card_index < 0 or board_index < 0:
            (card_index, board_index) = self.GreedyPlay(state, self.nn_output_normalize(state))
        return ( (state.left_cards + state.right_cards)[card_index], Helper.idx2tuple(board_index, gm.BOARD_SIZE) )
        
    def nn_output_normalize(self, state):
        [card_output, move_output] = self.forward(fe.state2feature(state, self.features))
        output = np.concatenate((card_output, move_output), axis=1)
        mask = np.reshape(state.get_unplayed_cards_by_index() + state.get_legal_moves_by_index(), (1, -1))
        return (output * mask)[0].flatten()
           
    def forward(self, input):
        return self.predict_func([input, 0])
    
    def fit(self, states, card_actions, move_actions, won):
        """
        The fit method will update the policy by a batch of simulated experiences
        Args: 
            states: n x dim x 10 array. n is the number of samples to be trained (steps player has played). dim is the dimension of all the features
            actions: n x (2 * hand_size + board_size ** 2 = 19) array. A one-hot array for each possible action
            rewards: n x 1 array. With either +1/0/-1 for win/tie/lose
        """
        self.model.optimizer.lr = K.abs(self.model.optimizer.lr) * (+1 if won else -1)
        # TODO: update to fit with epochs 
        self.model.train_on_batch(states, [card_actions, move_actions])
        
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
        self.features = object_specs['feature_list']
        if 'weights_file' in object_specs:
            model.load_weights(object_specs['weights_file'])
        return model
    
    def load_weights(self, weigfht_file):
        self.model.load_weights(weigfht_file)
    
    def GreedyPlay(self, state, action):
    # action is a vector of length 19 for the Probability Distribution of playing a card to a position, so action = card + move 
    # Card is one of the 10 cards, from left+right cards list. Move is one of the 9 cells on the board, left to right and top to bottom. 
    # So it looks like this for one-hot case: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0,    0, 0, 0, 1, 0, 0, 0, 0, 0]
    # Or like this for probabilistic case: [0.05, 0.15, 0, 0.05, 0.75, 0, 0, 0, 0, 0,    0.02, 0, 0.12, 0.68, 0, 0.08, 0.1, 0, 0]
    # This example means we will pick the 5th card from the left player's hands, and place on the (0, 1) cell on the board

        if len(action) != 2 * gm.START_HANDS + gm.BOARD_SIZE ** 2:
            raise ValueError("The action must have {} dimensions, but instead it has {} dimensions".format(2 * gm.START_HANDS + \
                gm.BOARD_SIZE ** 2, len(action)))
        card_index = np.argmax(action[:2 * gm.START_HANDS], axis = 0)
        board_index = np.argmax(action[2 * gm.START_HANDS:], axis = 0)
    
        return (card_index, board_index)
   
class Bias(Layer):
    """
    Custom keras layer that simply adds a scalar bias to each location in the input

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
    

