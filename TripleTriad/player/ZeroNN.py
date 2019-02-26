import TripleTriad.feature as fe
import TripleTriad.game as gm
from TripleTriad.player.policy import Policy
import TripleTriad.game_helper as Helper

import numpy as np
import copy
import json
import os

from keras.models import Sequential, Model, model_from_json, clone_model
from keras.layers import Input, BatchNormalization, Dense, MaxPooling1D, SeparableConv1D, merge
from keras.layers.core import Activation, Flatten, Dropout
from keras import backend as K


CONV_LAYER_PARAMETERS = {
    "filter": 256,
    "strides": 3
}

RESIDUAL_LAYER_PARAMETERS = {
    "filter": 256,
    "strides": 3,
    "layer_number": 10
}

POLICY_HEAD_PARAMETERS = {
    "filter": 2,
    "strides": 1,
    "plane_number": 45
}

VALUE_HEAD_PARAMETERS = {
    "filter": 1,
    "strides": 1,
    "hidden_size": 256
}


DEFAULT_MODEL_FILE = "model.json"


class ZeroPolicy(Policy):

    def __init__(self, model_save_path, features=fe.DEFAULT_FEATURES, model_load_path=None):
        self.features = features
        self.model_save_path = model_save_path
        self.model = None
        if model_load_path is not None:
            self.load_model(model_load_path)
        else:
            self.create_neural_network()
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

        state_input = Input(shape=(fe.get_feature_dim(self.features), 2 * gm.START_HANDS))

        # Testing for Conv1d layer

        x = SeparableConv1D(CONV_LAYER_PARAMETERS["filters"], CONV_LAYER_PARAMETERS["strides"], padding='same',
                            data_format='channels_first', name="Conv_kernal_1")(state_input)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        # x = MaxPooling1D(padding='same')(x)

        for i in range(RESIDUAL_LAYER_PARAMETERS["layer_number"]):
            x = self.residual_layer(x)

        # x = Flatten()(x)

        policy_output = self.policy_head(x)
        value_output = self.value_head(x)
        network = Model(input=state_input, output=[policy_output, value_output])
        self.model = network

    def policy_head(self, in_layer):
        ph = SeparableConv1D(POLICY_HEAD_PARAMETERS["filters"], POLICY_HEAD_PARAMETERS["strides"], padding='same',
                             data_format='channels_first', name="Policy_Head_Conv")(in_layer)
        ph = BatchNormalization()(ph)
        ph = Activation("relu")(ph)
        ph = Dense(POLICY_HEAD_PARAMETERS["plane_number"], name="Policy_head_dense")(ph)
        return ph

    def value_head(self, in_layer):
        vh = SeparableConv1D(VALUE_HEAD_PARAMETERS["filters"], VALUE_HEAD_PARAMETERS["strides"], padding='same',
                             data_format='channels_first', name="Value_Head_Conv")(in_layer)
        vh = BatchNormalization()(vh)
        vh = Activation("relu")(vh)
        vh = Dense(VALUE_HEAD_PARAMETERS["hidden_size"])(vh)
        vh = Activation("relu")(vh)
        vh = Dense(1)(vh)
        vh = Activation("tanh")(vh)
        return vh

    def residual_layer(self, in_layer):
        rl = SeparableConv1D(RESIDUAL_LAYER_PARAMETERS["filters"], RESIDUAL_LAYER_PARAMETERS["strides"], padding='same',
                             data_format='channels_first', name="Policy_Head_Conv")(in_layer)
        rl = BatchNormalization()(rl)
        rl = Activation("relu")(rl)
        rl = SeparableConv1D(RESIDUAL_LAYER_PARAMETERS["filters"], RESIDUAL_LAYER_PARAMETERS["strides"], padding='same',
                             data_format='channels_first', name="Policy_Head_Conv")(rl)
        rl = BatchNormalization()(rl)
        rl = merge([rl, in_layer], mode = 'sum') # Skip connection
        rl = Activation("relu")(rl)
        return rl

    def print_network(self):
        self.model.summary()

    def clone(self):
        new_policy = ZeroPolicy()
        new_policy.features = copy.copy(self.features)
        new_policy.model_save_path = self.model_save_path
        new_policy.model = clone_model(self.model)
        return new_policy

    def get_action(self, state, card_index=-1, board_index=-1):
        # This will return the (card, move) pair, where card is the Card object, and move is the (x, y) tuple
        # TODO: Change this to MCTS play
        if card_index < 0 or board_index < 0:
            (card_index, board_index) = self.GreedyPlay(state, self.nn_output_normalize(state))
        return ((state.left_cards + state.right_cards)[card_index], Helper.idx2tuple(board_index, gm.BOARD_SIZE))


    def get_MCTS_probs(self):
        pass

    def nn_output_normalize(self, state):
        # TODO: update to new action
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

    def save_model(self, model_file = DEFAULT_MODEL_FILE, weights_file=None):
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

        with open(os.path.join(self.model_save_path, model_file), 'w') as f:
            json.dump(object_specs, f)

    def load_model(self, model_file=DEFAULT_MODEL_FILE):
        """
        Load the neural network model from a json file in the self.model_load_path
        """
        with open(os.path.join(self.model_save_path, model_file), 'r') as f:
            object_specs = json.load(f)
        self.model = model_from_json(object_specs['keras_model'])
        self.features = object_specs['feature_list']
        if 'weights_file' in object_specs:
            self.model.load_weights(object_specs['weights_file'])

    def save_weights(self, weight_file):
        self.model.save_weights(os.path.join(self.model_save_path, weight_file))

    def load_weights(self, weigfht_file):
        self.model.load_weights(os.path.join(self.model_save_path, weigfht_file))


