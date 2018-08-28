import TripleTriad.feature as fe
import TripleTriad.game as gm
from TripleTriad.player.NNPolicy import NNPolicy
from TripleTriad.mcts import MCTS
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

class MCTSPolicy(NNPolicy):
    
    def __init__(self, params = DEFAULT_NN_PARAMETERS, features = fe.DEFAULT_FEATURES, model_save_path = DEFAULT_MODEL_OUTPUT_PATH, 
                 model_load_path = None):
        NNPolicy.__init__(self, params, features, model_save_path, model_load_path)
        self.mcts = MCTS()
            
    def clone(self):
        new_policy = MCTSPolicy()
        new_policy.params = self.params.copy()
        new_policy.features = copy.copy(self.features)
        new_policy.model_save_path = self.model_save_path
        new_policy.model = clone_model(self.model)
        return new_policy
        
    def get_action(self, state, competitive):
        # This will search all the action space with MCTS and return the optimal move after certain amount of simulations
        action_scores, action = self.mcts.search(self.board, player, competitive=competitive)

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
        