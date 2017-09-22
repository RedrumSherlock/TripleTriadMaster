from TripleTriad.feature import *
from TripleTriad.game import GameState
import unittest
import numpy as np
import random

class TestDefaultFeature(unittest.TestCase):
    
    def test_game_start(self):
        game = GameState()
        self.assertTrue(get_feature_dim() == 17)
        features = state2feature(game)
        self.assertTrue(features.shape[0] == 17)
        self.assertTrue(features.shape[1] == 2 * Game.START_HANDS)
        
        # At the beginning all the cards are in the hands equally
        self.assertTrue((np.sum(features[4:15, :], axis = 1) == [0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5]).all())
        # At the beginning all the cards are equally owner by both players
        self.assertTrue(np.sum(features[15, :]) == Game.START_HANDS)
        # The current player should either be 1 for all the cards, or 0 for all the cards
        start_player = np.sum( features[16, :])
        self.assertTrue(start_player == features.shape[1] or start_player == 0 )
      
    
    def test_game_end(self):
        game = GameState()  
        features = state2feature(game)
        start_player = np.sum( features[16, :])
        
        # Play the end until the end
        while(not game.is_end_of_game()):
            move = random.choice(game.get_legal_moves())
            card = random.choice(game.get_unplayed_cards())
            game.play_round(card, *move)
        features = state2feature(game)
         
        # At the end 9 cards should fill up the board
        self.assertTrue((np.sum(features[4:13, :], axis = 1) == [1, 1, 1, 1, 1, 1, 1, 1, 1]).all())
        # At the end the winner should own more card
        if game.get_winner() == Game.LEFT_PLAYER:
            self.assertTrue(np.sum(features[15, :]) > Game.START_HANDS)
        elif game.get_winner() == Game.RIGHT_PLAYER:
            self.assertTrue(np.sum(features[15, :]) < Game.START_HANDS)
        else:
            self.assertTrue(np.sum(features[15, :]) == Game.START_HANDS)
        # The current player should either be 1 for all the cards, or 0 for all the cards
        self.assertTrue(np.sum( features[16, :]) == features.shape[1] - start_player )     
           
if __name__ == '__main__':
    unittest.main()
