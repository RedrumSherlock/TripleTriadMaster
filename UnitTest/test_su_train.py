from TripleTriad.player.NNPolicy import NNPolicy
from TripleTriad.player.basic_policy import BaselinePolicy
import TripleTriad.training.su_train as su
import TripleTriad.game as gm
import TripleTriad.feature as fe
from TripleTriad.game_helper import timer 

import unittest
import warnings
import numpy as np

class TestMCTrainingProcess(unittest.TestCase):
    
    def setUp(self):
        self.single_meta = {
            "batch_size": 1,
            "num_wins": {},
            "out_directory": "test_cards",
            "card_path": "test_cards",
            "card_file": "cards.csv"
            }
        self.multi_meta = {
            "batch_size": 5,
            "num_wins": {},
            "out_directory": "test_cards",
            "card_path": "test_cards",
            "card_file": "cards.csv"
            }
        self.target = BaselinePolicy()
        
    def test_run_single_game(self):
        
        game = gm.GameState()
        (states, cards, moves) = su.simulate_single_game(self.target, game)
        self.assertTrue(np.array(states).shape == (gm.BOARD_SIZE **2, fe.get_feature_dim(), gm.START_HANDS * 2))
        self.assertTrue(np.array(cards).shape == (gm.BOARD_SIZE **2, gm.START_HANDS * 2))
        self.assertTrue(np.array(moves).shape == (gm.BOARD_SIZE **2, gm.BOARD_SIZE **2))
        sum_cards = np.sum(np.array(cards), 0)
        sum_moves = np.sum(np.array(moves), 0)
        self.assertTrue( np.all( sum_cards <= 1) )
        self.assertTrue( np.all( sum_moves == 1) )
        
    @timer
    def test_generator(self):
        data_generator = su.state_action_generator(self.target, self.multi_meta)
        for _ in range(1):
            (states, action) = next(data_generator)
            self.assertTrue(states.shape == (self.multi_meta["batch_size"] * (gm.BOARD_SIZE **2), fe.get_feature_dim(), gm.START_HANDS * 2))
            self.assertTrue(action["card_output"].shape == (self.multi_meta["batch_size"] * (gm.BOARD_SIZE **2), gm.START_HANDS * 2 ))
            self.assertTrue(action["move_output"].shape == (self.multi_meta["batch_size"] * (gm.BOARD_SIZE **2), gm.BOARD_SIZE ** 2))
            
if __name__ == '__main__':
    unittest.main()
