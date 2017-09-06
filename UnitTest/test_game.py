from TripleTriad.game import GameState
import unittest

class TestGame(unittest.TestCase):
    
    def test_fullgame(self):
        game = GameState()
        self.assertTrue(len(game.left_cards) == GameState.START_HANDS)
        self.assertTrue(len(game.right_cards) == GameState.START_HANDS)
    
    
if __name__ == '__main__':
    unittest.main()
