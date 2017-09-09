from TripleTriad.game import *
import unittest
import random
import copy

class TestGame(unittest.TestCase):
    
    def test_fullgame(self):
        game = GameState()
        self.assertTrue(len(game.left_cards) == game.start_hands)
        self.assertTrue(len(game.right_cards) == game.start_hands)
        while(not game.is_end_of_game()):
            move = random.choice(game.get_legal_moves())
            card = random.choice(game.get_unplayed_cards())
            game.play_round(card, *move)
            game.print_board()
        
    def test_random_games(self):
        
        default_left_cards = load_cards_from_file("", "cards.csv")
        default_right_cards = load_cards_from_file("", "cards.csv")
        
        winner = []
        for _ in range(10000):
            left_cards = random.sample(default_left_cards, 5)
            right_cards = random.sample(default_right_cards, 5)
            
            for card in left_cards + right_cards:
                card.reset()
                
            game = GameState(left_cards = left_cards, right_cards = right_cards)
            self.assertTrue(len(game.left_cards) == game.start_hands)
            self.assertTrue(len(game.right_cards) == game.start_hands)
            while(not game.is_end_of_game()):
                move = random.choice(game.get_legal_moves())
                card = random.choice(game.get_unplayed_cards())
                game.play_round(card, *move)
            
            self.assertTrue(game.get_winner() is not None)
            winner.append(game.get_winner())
        print 'Winners:'
        print 'Left', 'Right', 'Tie'    
        print len(filter(lambda x: x == 1, winner)), len(filter(lambda x: x == -1, winner)), len(filter(lambda x: x == 0, winner))
    
if __name__ == '__main__':
    unittest.main()
