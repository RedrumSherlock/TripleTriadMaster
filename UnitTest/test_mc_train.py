from TripleTriad.mc_train import *
import unittest

class TestMCTrainingProcess(unittest.TestCase):
    
    def test_nn_policy(self):
        game = GameState()
        feature_dim, features = state2feature(game, ["board_numbers"])
        self.assertTrue(feature_dim == len(features))
            
if __name__ == '__main__':
    unittest.main()
