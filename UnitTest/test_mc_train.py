from TripleTriad.mc_train import *
from TripleTriad.policy import NNPolicy
from keras.optimizers import SGD
import unittest

class TestMCTrainingProcess(unittest.TestCase):
    
    def test_run_single_game(self):
        single_meta = {
            "game_batch": 1,
            "win_ratio": {},
            "out_directory": "test_cards",
            "card_set": "cards.csv"
            }
        player = NNPolicy()
        opponent = player.clone()
        
        (states, actions, rewards) = simulate_games(player, opponent, single_meta)
        self.assertTrue(len(states) == 1 and len(actions) == 1 and len(rewards) == 1)
        self.assertTrue(len(states[0]) == 4 or len(states[0]) == 5)
        self.assertTrue(len(actions[0]) == len(states[0]) and len(rewards[0]) == len(states[0]))
        
    def test_train_single_game(self):
        single_meta = {
            "game_batch": 1,
            "win_ratio": {},
            "out_directory": "test_cards",
            "card_set": "cards.csv",
            "learning_rate": 0.001
            }
        player = NNPolicy()
        opponent = player.clone()
        optimizer = SGD(lr=single_meta["learning_rate"])
        player.model.compile(loss=log_loss, optimizer=optimizer)
        (states, actions, rewards) = simulate_games(player, opponent, single_meta)   
        train_on_results(player, states, actions, rewards)
            
if __name__ == '__main__':
    unittest.main()
