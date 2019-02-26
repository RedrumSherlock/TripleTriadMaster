from TripleTriad.player.policy import RandomPolicy
from TripleTriad.player.basic_policy import BasicPolicy, BaselinePolicy
from TripleTriad.player.NNPolicy import NNPolicy
from TripleTriad.game_helper import timer
import TripleTriad.game as gm
from TripleTriad.training.mc_train import ZEROTH_FILE

import random
import os
import json


def evaluate_nn_policy():
    """
    To evaluate the results gained from the training process. It can be run in parallel when the training is happening.
    There should be a metadata.json file for the metadata of training process, a model.json for the model trained, 
    and at least one weights.%05d.hdf5 weight file in the output directory from the training process.
    """
    import argparse
    parser = argparse.ArgumentParser(description='Compare the trained NN policy to our manually crafted baseline policy')
    parser.add_argument("directory", help="Path to folder where the model params and metadata was saved from training.")
    parser.add_argument("--metadata-file", help="The meta data file to be loaded", default="su_metadata.json")
    parser.add_argument("--weight-file", help="The weight file to be loaded to the model", default=ZEROTH_FILE)
    parser.add_argument("--plot", help="Plot the evaluation results", default=True, action="store_true")
    parser.add_argument("--num-games", help="Number of games to play for evaluation", type=int, default=1000)
    parser.add_argument("--card-path", help="The directory with the card set file (Default: {})".format(gm.DEFAULT_PATH), default=gm.DEFAULT_PATH)
    parser.add_argument("--card-file", help="The file containing the cards to play with (Default: {})".format(gm.DEFAULT_CARDS_FILE), default=gm.DEFAULT_CARDS_FILE)
    parser.add_argument("--verbose", "-v", help="Turn on verbose mode", default=True, action="store_true")
    args = parser.parse_args()
    
    with open(os.path.join(args.directory, args.metadata_file), "r") as f:
        metadata = json.load(f)
    
    with open(os.path.join(args.directory, metadata["model_file"]), "r") as f:
        player = NNPolicy(model_load_path = os.path.join(args.directory, metadata["model_file"]))
    
    player.load_weights(os.path.join(args.directory, args.weight_file))
    opponent = BaselinePolicy()
    compare_policy(player, opponent, args.num_games, args.card_path, args.card_file)


def evaluate_basic_policy():
    player = BasicPolicy()
    opponent = RandomPolicy()
    compare_policy(player, opponent, 1000)


@timer
def compare_policy(player, opponent, num_games, card_file_path = "test_cards", card_file_name = "cards.csv"):
    default_left_cards = gm.GameState.load_cards_from_file(card_file_path, card_file_name)
    default_right_cards = gm.GameState.load_cards_from_file(card_file_path, card_file_name)
    
    winner = []
    for i in range(num_games):
        left_cards = random.sample(default_left_cards, 5)
        right_cards = random.sample(default_right_cards, 5)
        
        for card in left_cards + right_cards:
            card.reset()
            
        game = gm.GameState(left_cards = left_cards, right_cards = right_cards)
        while not game.is_end_of_game():
            # Player is always on the left, and the opponent is always on the right. Randomly picks who starts the game.
            if game.current_player == gm.LEFT_PLAYER:
                (card, move) = player.get_action(game)
            else:
                (card, move) = opponent.get_action(game)
            game.play_round(card, *move)
        
        winner.append(game.get_winner())
        """
        if i%10 == 0 and i > 0:
            won_games = sum(1 for _ in filter(lambda x: x == gm.LEFT_PLAYER, winner))
            tie_games = sum(1 for _ in filter(lambda x: x== gm.NO_ONE, winner))
            lost_games = sum(1 for _ in filter(lambda x: x== gm.RIGHT_PLAYER, winner))
            print("This is the {}th game, current win rate: {}, tie rate: {}, lose rate: {}".format(i, round(won_games / i, 2), \
                                                                          round(tie_games / i, 2), round(lost_games / i, 2)), end='\r')
        """

    won_games = sum(1 for _ in filter(lambda x: x == gm.LEFT_PLAYER, winner))
    tie_games = sum(1 for _ in filter(lambda x: x == gm.NO_ONE, winner))
    lost_games = sum(1 for _ in filter(lambda x: x == gm.RIGHT_PLAYER, winner))
    print("Evaluation done. Player won {} games, tied {} games, and lost {} games".format(won_games, tie_games, lost_games))
    return round(won_games / num_games, 2)


if __name__ == '__main__':
    evaluate_nn_policy()