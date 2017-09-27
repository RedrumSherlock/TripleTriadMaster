from TripleTriad.mc_train import *
from TripleTriad.policy import NNPolicy
import unittest

class TestMCTrainingProcess(unittest.TestCase):
    
    def test_single_game_fit(self):
        player = NNPolicy()
        opp = player.clone()
            
if __name__ == '__main__':
    unittest.main()
