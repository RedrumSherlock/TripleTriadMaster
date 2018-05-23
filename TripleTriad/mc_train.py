import os
import json
import re
import numpy as np
import random
from shutil import copyfile
from keras.optimizers import SGD
import keras.backend as K
from TripleTriad.game import GameState
import TripleTriad.game as Game
from TripleTriad.policy import *
import TripleTriad.feature as FE
import TripleTriad.game_helper as Helper

ZEROTH_FILE = "weights.00000.hdf5"
 
def log_loss(y_true, y_pred):
    '''Keras 'loss' function for the REINFORCE algorithm, where y_true is the action that was
    taken, and updates with the negative gradient will make that action more likely. We use the
    negative gradient because keras expects training data to minimize a loss function.
    '''
    return -y_true * K.log(K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon()))


def save_metadata(metadata, directory, filename):
    with open(os.path.join(directory, filename), "w") as f:
        json.dump(metadata, f, sort_keys=True, indent=2)
            
def simulate_games(player, opponent, metadata):
    """
    Args:
        player: a policy for the player side
        opponent: another policy for the opponent side
        metadata: a dictionary which contains the meta data for this training process
        
    Returns:
        states: a list with n elements, where n is the number of games (in each batch) specified by game_batch in metadata. Each element is another list
                with m elements, where m is the moves made in this game by the player (we only train based on the player actions, not the opponent). Each 
                element in this list is the game feature (basically a ndarray with size 1xDIMx10. DIM is the dimension of all selected features for each card)
        actions: Similar to the states, a list of list for each game, and the element in the inner list is a one-hot vector representing the action (a 1xn 
                ndarray where n=2*HAND_SIZE + BOARD_SIZE**2. The number one in this array represents the move, i.e. from a hand index to a board index)
        rewards: Similar to the actions, a list of list for each game, and the element in the inner list is a number, either 1 or 0 which respresent win 
                or lose for the whole game. 
        
    """
    
    states = [[] for _ in range(metadata["game_batch"])] # Feature from the game state, i.e. by default feature a 17 x 10 array
    actions = [[] for _ in range(metadata["game_batch"])] # Each action should be a pair of (move, card). Move is the one-hot vector for 9 cells on the board. 
                                                          # Card is the one-hot vector for the 10 cards from left to right (must be cards not added to board yet) 
    rewards = [[] for _ in range(metadata["game_batch"])] # Either player has won (1), tied (0), or lost (-1)
    
    # Learner is always the left player, and the opponent picked from the pool is always the right player
    # Game will start randomly by left or right player by a 50/50
    left_card_set = Game.load_cards_from_file(metadata["out_directory"], metadata["card_set"])
    right_card_set = Game.load_cards_from_file(metadata["out_directory"], metadata["card_set"])
    
    for i in range(metadata["game_batch"]):
        left_cards = random.sample(left_card_set, Game.START_HANDS)
        right_cards = random.sample(right_card_set, Game.START_HANDS)
            
        new_game = GameState(left_cards = left_cards, right_cards = right_cards)
        
        while(not new_game.is_end_of_game()):
            if new_game.current_player == Game.LEFT_PLAYER:
                # Record all the moves made by the learner
                (card_index, board_index) = GreedyPlay(new_game, player.nn_output_normalize(new_game))
                action = Helper.indices2onehot(card_index, board_index, Game.BOARD_SIZE, Game.START_HANDS)
                (card, move) = player.get_action(new_game, card_index, board_index)
                states[i].append(FE.state2feature(new_game))
                actions[i].append(action)
                rewards[i].append(1)
            else:
                (card, move) = opponent.get_action(new_game)
            new_game.play_round(card, *move)
        
        rewards[i] = np.dot(rewards[i], new_game.get_winner()).tolist()
        
    return (states, actions, rewards)

def train_on_results(policy, states, actions, rewards):
    for (st_tensor, mv_tensor, won) in zip(states, actions, rewards):
        policy.fit(np.concatenate(st_tensor, axis=0), 
                   np.concatenate(mv_tensor, axis=0), 
                   won)
            
def run_training(cmd_line_args=None):
    import argparse
    parser = argparse.ArgumentParser(description='Train the policy network with Monte Carlo approach and exploring start')
    parser.add_argument("out_directory", help="Path to folder where the model params and metadata will be saved after each epoch.")
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
    
    # Baseline function (TODO) default lambda state: 0  (receives either file
    # paths to JSON and weights or None, in which case it uses default baseline 0)
    if cmd_line_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(cmd_line_args)

   

    if not os.path.exists(args.out_directory):
        if args.verbose:
            print("creating output directory {}".format(args.out_directory))
        os.makedirs(args.out_directory)
    
    
    if not os.path.exists(os.path.join(args.out_directory, args.card_set)):
        raise ValueError("Cannot resume without card file {} in the output directory".format(args.card_set))
            
                
    if not args.resume:
        # starting the game from scratch
        metadata = {
            "out_directory": args.out_directory,
            "model_file": args.model_json,
            "init_weights": args.initial_weights,
            "learning_rate": args.learning_rate,
            "game_batch": args.game_batch,
            "save_every": args.save_every,
            "card_set": args.card_set,
            "opponents": [ZEROTH_FILE],  # which weights from which to sample an opponent each batch
            "num_wins": {}  # number of wins for player in each batch
        }
        player_weights = ZEROTH_FILE
        iter_start = 1
        player = NNPolicy()
        save_metadata(metadata, args.out_directory, "metadata.json")
        # Create the Zeroth weight file
        player.model.save_weights(os.path.join(args.out_directory, player_weights))
    else:
        # Load the metadata
        if not os.path.exists(os.path.join(args.out_directory, "metadata.json")):
            raise ValueError("Cannot resume without metadata.json file in the output directory")
        with open(os.path.join(args.out_directory, "metadata.json"), "r") as f:
            metadata = json.load(f)
        
        # Load the model    
        if not os.path.exists(args.model_json):
            raise ValueError("Cannot resume without model json file in the output directory")
        args.model_json = os.path.join(args.out_directory, os.path.basename(args.model_json))
        with open(args.model_json, 'r') as f:
            object_specs = json.load(f)
        if 'weights_file' in object_specs and not os.path.exists(os.path.join(args.out_directory, os.path.basename(object_specs['weights_file']))):
            raise ValueError("The weight file {} specified by the model json file is not existing in the output directory".format(os.path.basename(object_specs['weights_file'])))                                        
        
        if args.verbose:
            print("Resuming with model {}".format(args.model_json))
        player = NNPolicy(model_load_path = args.model_json)
        
        # Load the initial weights
        if not re.match(r"weights\.\d{5}\.hdf5", args.initial_weights):
            raise ValueError("Expected to resume from weights file with name 'weights.#####.hdf5'")
        player_weights = args.initial_weights
        args.initial_weights = os.path.join(args.out_directory, os.path.basename(args.initial_weights))
        if not os.path.exists(args.initial_weights):
            raise ValueError("Cannot resume without weight file {} in the output directory".format(args.initial_weights))

        if args.verbose:
            print("Resuming with weights {}".format(args.initial_weights))
        player.model.load_weights(args.initial_weights)    
        iter_start = 1 + int(player_weights[8:13])
        
        

    opponent = player.clone()

    if args.verbose:
        print("created player and opponent")


    # Append args of current run to history of full command args.
    metadata["cmd_line_args"] = metadata.get("cmd_line_args", [])
    metadata["cmd_line_args"].append(vars(args))

    optimizer = SGD(lr=args.learning_rate)
    player.model.compile(loss=log_loss, optimizer=optimizer)
    for i_iter in range(iter_start, args.iterations + 1):
        # Note that player_weights will only be saved as a file every args.record_every iterations.
        # Regardless, player_weights enters into the metadata to keep track of the win ratio over
        # time.
        player_weights = "weights.%05d.hdf5" % i_iter

        # Randomly choose opponent from pool (possibly self), and playing
        # game_batch games against them.
        opp_weights = np.random.choice(metadata["opponents"])
        opp_path = os.path.join(args.out_directory, opp_weights)

        # Load new weights into opponent's network, but keep the same opponent object.
        opponent.model.load_weights(opp_path)
        if args.verbose:
            print("Batch {}\tsampled opponent is {}".format(i_iter, opp_weights))

        # Run games (and learn from results)
        (states, actions, rewards) = simulate_games(player, opponent, metadata)
        train_on_results(player, states, actions, rewards)
        metadata["num_wins"][player_weights] = sum(reward[0] == 1 for reward in rewards)

        # Save intermediate models.
        if i_iter % args.record_every == 0:
            player.model.save_weights(os.path.join(args.out_directory, player_weights))

        # Add player to batch of oppenents once in a while.
        if i_iter % args.save_every == 0:
            metadata["opponents"].append(player_weights)
        save_metadata(metadata, args.out_directory, "metadata.json")


if __name__ == '__main__':
    run_training()