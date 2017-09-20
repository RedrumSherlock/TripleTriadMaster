from TripleTriad.feature import *
from TripleTriad.game import GameState
import unittest
import numpy as np

class TestFeature(unittest.TestCase):
    
    def test_default_feature(self):
        game = GameState()
        self.assertTrue(get_feature_dim() == 17)
        features = state2feature(game)
        self.assertTrue(features.shape[0] == 17)
        self.assertTrue(features.shape[1] == 2 * Game.START_HANDS)
            
if __name__ == '__main__':
    unittest.main()
