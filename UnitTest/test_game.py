from TripleTriad.game import *
from TripleTriad.policy import RandomPolicy
import unittest
import copy

class TestGame(unittest.TestCase):
    
    def test_fullgame(self):
        game = GameState()
        self.assertTrue(len(game.left_cards) == game.start_hands)
        self.assertTrue(len(game.right_cards) == game.start_hands)
        turns = 0
        player = RandomPolicy()
        while(not game.is_end_of_game()):
            (card, move) = player.get_action(game)
            game.play_round(card, *move)
            turns += 1
        self.assertTrue(sum( 1 for _ in filter(lambda x: x is None, game.board)) == 0)
        self.assertTrue(turns == game.board_size * game.board_size)
        
    def test_random_games(self):
        
        default_left_cards = load_cards_from_file("test_cards", "cards.csv")
        default_right_cards = load_cards_from_file("test_cards", "cards.csv")
        
        winner = []
        player = RandomPolicy()
        iter = 10000
        for _ in range(iter):
            left_cards = random.sample(default_left_cards, 5)
            right_cards = random.sample(default_right_cards, 5)
            
            for card in left_cards + right_cards:
                card.reset()
                
            game = GameState(left_cards = left_cards, right_cards = right_cards)
            self.assertTrue(len(game.left_cards) == game.start_hands)
            self.assertTrue(len(game.right_cards) == game.start_hands)
            while(not game.is_end_of_game()):
                (card, move) = player.get_action(game)
                game.play_round(card, *move)
            
            self.assertTrue(game.get_winner() is not None)
            winner.append(game.get_winner())
        # when we play as random vs random and play enough times with basic rules, the results should be equally distributed across win, lose, and tie
        # The chance that after enough game plays, ther number of games either player win is less than 10 percent of the total games is low enough to not be considered 
        self.assertTrue(sum( 1 for _ in filter(lambda x: x == 1, winner)) > iter/10 \
                        and sum( 1 for _ in filter(lambda x: x == -1, winner)) > iter/10 \
                        and sum( 1 for _ in filter(lambda x: x == 0, winner)) > iter/10)
    

if __name__ == '__main__':
    unittest.main()
