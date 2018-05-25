from TripleTriad.player.policy import RandomPolicy
from TripleTriad.player.basic_policy import BasicPolicy, BaselinePolicy
from TripleTriad.game_helper import timer
import TripleTriad.game as gm

import random


def evaluate_nn_policy():
    """
    To evaluate the training process
    """
    import argparse
    parser = argparse.ArgumentParser(description='Compare the trained NN policy to our manually crafted baseline policy')
    parser.add_argument("directory", help="Path to folder where the model params and metadata was saved from training.")
    parser.add_argument("--resume", help="Load latest weights in out_directory and resume", default=False, action="store_true")
    parser.add_argument("--model-json", help="JSON file for policy model in the output directory.", default = "model.json")
    parser.add_argument("--initial-weights", help="Path to HDF5 file with inital weights (i.e. result of supervised training).", default = ZEROTH_FILE)
    parser.add_argument("--learning-rate", help="Keras learning rate (Default: 0.001)", type=float, default=0.001)
    parser.add_argument("--save-every", help="Save policy as a new opponent every n batches (Default: 500)", type=int, default=500)
    parser.add_argument("--record-every", help="Save learner's weights every n batches (Default: 1)", type=int, default=1)
    parser.add_argument("--game-batch", help="Number of games per mini-batch (Default: 20)", type=int, default=20)
    parser.add_argument("--iterations", help="Number of training batches/iterations (Default: 10000)", type=int, default=10000)
    parser.add_argument("--card-set", help="Number of training batches/iterations (Default: {})".format(Game.DEFAULT_CARDS_FILE), default=Game.DEFAULT_CARDS_FILE)
    parser.add_argument("--verbose", "-v", help="Turn on verbose mode", default=True, action="store_true")

def evaluate_basic_policy():
    player = BasicPolicy()
    opponent = RandomPolicy()
    compare_policy(player, opponent, 1000)

@timer
def compare_policy(player, opponent, num_games, card_file_path = "test_cards", card_file_name = "cards.csv"):
    default_left_cards = gm.GameState.load_cards_from_file(card_file_path, card_file_name)
    default_right_cards = gm.GameState.load_cards_from_file(card_file_path, card_file_name)
    
    winner = []
    for _ in range(num_games):
        left_cards = random.sample(default_left_cards, 5)
        right_cards = random.sample(default_right_cards, 5)
        
        for card in left_cards + right_cards:
            card.reset()
            
        game = gm.GameState(left_cards = left_cards, right_cards = right_cards)
        while(not game.is_end_of_game()):
            # Player is always on the left, and the opponent is always on the right. Randomly picks who starts the game.
            if game.current_player == gm.LEFT_PLAYER:
                (card, move) = player.get_action(game)
            else:
                (card, move) = opponent.get_action(game)
            game.play_round(card, *move)
        
        winner.append(game.get_winner())
    
    print("Player won {} games, tied {} games, and lost {} games".format(sum(1 for _ in filter(lambda x: x == gm.LEFT_PLAYER, winner)), \
                                                                         sum(1 for _ in filter(lambda x: x== gm.NO_ONE, winner)), \
                                                                         sum(1 for _ in filter(lambda x: x== gm.RIGHT_PLAYER, winner))))
    
if __name__ == '__main__':
    evaluate_basic_policy()