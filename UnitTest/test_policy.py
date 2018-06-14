from TripleTriad.player.NNPolicy import NNPolicy
from TripleTriad.player.basic_policy import BaselinePolicy
import TripleTriad.game as gm
import TripleTriad.game_helper as Helper
import TripleTriad.feature as fe

import unittest
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout

class TestPolicy(unittest.TestCase):
    
    def test_nn_weights(self):
        player = NNPolicy()
        #self.assertTrue(player.model.get_weights()[0].shape == tuple(reversed((player.params["units"], 2 * gm.START_HANDS))))
        player.print_network()
        
    def test_get_action(self):
        player = NNPolicy()
        input = np.zeros((1, fe.get_feature_dim(player.features), 2 * gm.START_HANDS))
        
        game = gm.GameState()
        while(not game.is_end_of_game()): 
            (card, move) = player.get_action(game)
            self.assertTrue(card.position == (-1, -1) and card.owner == game.current_player)
            self.assertTrue(game.board[Helper.tuple2idx(game.board_size, *move)] is None)
            game.play_round(card, *move)
    
    def test_action_to_vector(self):
        player = BaselinePolicy()
        cards = []
        moves = []
        game = gm.GameState()
        while(not game.is_end_of_game()): 
            (card, move) = player.get_action(game)
            (card_vector, move_vector) = player.action_to_vector(game, card, move)
            self.assertTrue( sum(card_vector) == 1 and sum(move_vector) == 1)
            cards.append(card_vector)
            moves.append(move_vector)
            game.play_round(card, *move)
        sum_cards = np.sum(np.array(cards), 0)
        sum_moves = np.sum(np.array(moves), 0)
        self.assertTrue( np.all( sum_cards <= 1) )
        self.assertTrue( np.all( sum_moves == 1) )
         
    def test_save_model(self):
        player = NNPolicy(model_save_path = '/home/mike/tmp/test_nn_model.json')
        player.save_model()
        
    def test_load_model(self):
        player = NNPolicy()
        player.load_model('/home/mike/tmp/test_nn_model.json')
        
    def test_load_weights(self):
        player = NNPolicy(model_save_path = '/home/mike/tmp/test_nn_model_with_weights.json')
        player.save_model(weights_file='/home/mike/tmp/test_weight.hdf5')
        new_player = NNPolicy(model_load_path = '/home/mike/tmp/test_nn_model_with_weights.json')
        
        self.assertTrue(len(player.model.get_weights()) == len(new_player.model.get_weights()))
        for i in range(len(player.model.get_weights())):
            self.assertTrue( np.array_equal(player.model.get_weights()[i], new_player.model.get_weights()[i]) )
        
if __name__ == '__main__':
    unittest.main()
