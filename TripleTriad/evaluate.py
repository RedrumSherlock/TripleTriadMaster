from TripleTriad.policy import RandomPolicy
from TripleTriad.basic_policy import BasicPolicy
import TripleTriad.game as gm
import random


def evaluate():
    player = RandomPolicy()
    opponent = RandomPolicy()
    compare_policy(player, opponent, 1000)

def compare_policy(player, opponent, num_games):
    default_left_cards = gm.load_cards_from_file("test_cards", "cards.csv")
    default_right_cards = gm.load_cards_from_file("test_cards", "cards.csv")
    
    winner = []
    for _ in range(num_games):
        left_cards = random.sample(default_left_cards, 5)
        right_cards = random.sample(default_right_cards, 5)
        
        for card in left_cards + right_cards:
            card.reset()
            
        game = gm.GameState(left_cards = left_cards, right_cards = right_cards)
        while(not game.is_end_of_game()):
            if game.current_player == gm.LEFT_PLAYER:
                (card, move) = player.get_action(game)
            else:
                (card, move) = opponent.get_action(game)
            game.play_round(card, *move)
        
        winner.append(game.get_winner())
    
    print("Player won {} games, tied {} games, and lost {} games".format(sum(1 for _ in filter(lambda x: x==1, winner)), \
                                                                         sum(1 for _ in filter(lambda x: x==0, winner)), \
                                                                         sum(1 for _ in filter(lambda x: x==-1, winner))))
    
if __name__ == '__main__':
    evaluate()