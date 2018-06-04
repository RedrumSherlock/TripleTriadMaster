from TripleTriad.player.NNPolicy import NNPolicy
import TripleTriad.training.mc_train as mc

import unittest
import warnings

from keras.optimizers import SGD


class TestMCTrainingProcess(unittest.TestCase):
    
    def test_run_single_game(self):
        single_meta = {
            "game_batch": 1,
            "num_wins": {},
            "out_directory": "test_cards",
            "card_path": "test_cards",
            "card_file": "cards.csv"
            }
        player = NNPolicy()
        opponent = player.clone()
        
        (states, actions, rewards) = mc.simulate_games(player, opponent, single_meta)
        self.assertTrue(len(states) == 1 and len(actions) == 1 and len(rewards) == 1)
        self.assertTrue(len(states[0]) == 4 or len(states[0]) == 5)
        self.assertTrue(len(actions[0]) == len(states[0]) and len(rewards[0]) == len(states[0]))
        
    def test_train_single_game(self):
        single_meta = {
            "game_batch": 1,
            "num_wins": {},
            "out_directory": "test_cards",
            "card_path": "test_cards",
            "card_file": "cards.csv",
            "learning_rate": 0.001
            }
        player = NNPolicy()
        opponent = player.clone()
        optimizer = SGD(lr=single_meta["learning_rate"])
        player.model.compile(loss=mc.log_loss, optimizer=optimizer)
        (states, actions, rewards) = mc.simulate_games(player, opponent, single_meta)   
        mc.train_on_results(player, states, actions, rewards)
        
    def test_train_multi_games(self):
        num_games_batch = 20
        multi_meta = {
            "game_batch": num_games_batch,
            "num_wins": {},
            "out_directory": "test_cards",
            "card_path": "test_cards",
            "card_file": "cards.csv",
            "learning_rate": 0.001
            }
        player = NNPolicy()
        opponent = player.clone()
        
        (states, actions, rewards) = mc.simulate_games(player, opponent, multi_meta)
        
        self.assertTrue(len(states) == num_games_batch and len(actions) == num_games_batch and len(rewards) == num_games_batch)
        # Ensure both player got almost equal chance of playing first
        games_first = sum(len(state) == 5 for state in states)
        games_second = sum(len(state) == 4 for state in states)
        self.assertTrue(games_first + games_second == num_games_batch)
        if games_first == 0 or games_second == 0 :
            warnings.warn('Abnormal results: {} games first, {} games second'.format(games_first, games_second))
            
        self.assertTrue(len(actions[num_games_batch - 1]) == len(states[num_games_batch - 1]) and len(rewards[num_games_batch - 1]) == len(states[num_games_batch - 1]))
        games_won = sum(reward[0] == 1 for reward in rewards)
        games_tie = sum(reward[0] == 0 for reward in rewards)
        games_lost = sum(reward[0] == -1 for reward in rewards)
        self.assertTrue(games_won + games_tie + games_lost == num_games_batch)
        if games_won == 0 or games_tie == 0 or games_lost == 0:
            warnings.warn('Abnormal results: {} games won, {} games tie, {} lost'.format(games_won, games_tie, games_lost))
        
        optimizer = SGD(lr=multi_meta["learning_rate"])
        player.model.compile(loss=mc.log_loss, optimizer=optimizer)
        mc.train_on_results(player, states, actions, rewards)

            
if __name__ == '__main__':
    unittest.main()
