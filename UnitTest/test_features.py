from TripleTriad.feature import *
from TripleTriad.game import GameState
import unittest

class TestFeature(unittest.TestCase):
    
    def test_one_feature(self):
        game = GameState()
        feature_dim, features = state2feature(game, ["board_numbers"])
        self.assertTrue(feature_dim == len(features))
            
if __name__ == '__main__':
    unittest.main()
