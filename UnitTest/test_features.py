from TripleTriad.feature import *
from TripleTriad.game import GameState
from TripleTriad.player.policy import RandomPolicy
import unittest
import numpy as np
import random

class TestDefaultFeature(unittest.TestCase):
    
    def test_game_start(self):
        game = GameState()
        self.assertTrue(get_feature_dim() == 17)
        features = state2feature(game)
        self.assertTrue(features.shape[0] == 1)
        self.assertTrue(features.shape[1] == 17)
        self.assertTrue(features.shape[2] == 2 * Game.START_HANDS)
        
        # At the beginning all the cards are in the hands equally
        self.assertTrue((np.sum(features[0, 4:15, :], axis = 1) == [0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5]).all())
        # At the beginning all the cards are equally owner by both players
        self.assertTrue(np.sum(features[0, 15, :]) == Game.START_HANDS)
        # The current player should either be 1 for all the cards, or 0 for all the cards
        start_player = np.sum( features[0, 16, :])
        self.assertTrue(start_player == features.shape[2] or start_player == 0 )
      
    
    def test_game_end(self):
        game = GameState()  
        features = state2feature(game)
        start_player = np.sum( features[0, 16, :])
        player = RandomPolicy()
        
        # Play the end until the end
        while(not game.is_end_of_game()):
            (card, move) = player.get_action(game)
            game.play_round(card, *move)
        features = state2feature(game)
         
        # At the end 9 cards should fill up the board
        self.assertTrue((np.sum(features[0, 4:13, :], axis = 1) == [1, 1, 1, 1, 1, 1, 1, 1, 1]).all())
        # At the end the winner should own more card
        if game.get_winner() == Game.LEFT_PLAYER:
            self.assertTrue(np.sum(features[0, 15, :]) > Game.START_HANDS)
        elif game.get_winner() == Game.RIGHT_PLAYER:
            self.assertTrue(np.sum(features[0, 15, :]) < Game.START_HANDS)
        else:
            self.assertTrue(np.sum(features[0, 15, :]) == Game.START_HANDS)
        # The current player should either be 1 for all the cards, or 0 for all the cards
        self.assertTrue(np.sum( features[0, 16, :]) == features.shape[2] - start_player )     
           
if __name__ == '__main__':
    unittest.main()
