import TripleTriad.game as gm
from TripleTriad.player.NNPolicy import NNPolicy
import TripleTriad.feature as fe
import TripleTriad.game_helper as Helper

import os
import json
import re
import numpy as np
import random
from shutil import copyfile

from keras.optimizers import SGD
import keras.backend as K


ZEROTH_FILE = "weights.00000.hdf5"
 
def log_loss(y_true, y_pred):
    '''Keras 'loss' function for the REINFORCE algorithm, where y_true is the action that was
    taken, and updates with the negative gradient will make that action more likely. We use the
    negative gradient because keras expects training data to minimize a loss function.
    '''
    return -y_true * K.log(K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon()))

def pg_loss(y_true, y_pred):
    '''Keras 'loss' function for the REINFORCE algorithm, where y_true is the action that was
    taken, and updates with the negative gradient will make that action more likely. We use the
    negative gradient because keras expects training data to minimize a loss function.
    '''
    return K.sum(K.log(K.sum(y_true * K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon()), axis=1)))

                  
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
        card_actions: Similar to the states, a list of list for each game, and the element in the inner list is a one-hot vector representing the action 
                for picking a card(a 1xn ndarray where n=2*HAND_SIZE. The number one in this array represents the card to pick)
        move_actions: Similar to the states, a list of list for each game, and the element in the inner list is a one-hot vector representing the action 
                for picking a move(a 1xn ndarray where n=BOARD_SIZE**2. The number one in this array represents which grid on the board index to place the card picked)
        rewards: Similar to the actions, a list of list for each game, and the element in the inner list is a number, either 1 or 0 which represent win 
                or lose for the whole game. 
        
    """
    
    states = [[] for _ in range(metadata["game_batch"])] # Feature from the game state, i.e. by default feature a 16 x 10 array
    card_actions = [[] for _ in range(metadata["game_batch"])] # Card is the one-hot vector for the 10 cards from left to right 
    move_actions = [[] for _ in range(metadata["game_batch"])] # Move is the one-hot vector for the 9 possible moves
    rewards = [0 for _ in range(metadata["game_batch"])] # Either player has won (1), tied (0), or lost (-1)
    
    # Learner is always the left player, and the opponent picked from the pool is always the right player
    # Game will start randomly by left or right player by a 50/50
    left_card_file = gm.GameState.load_cards_from_file(metadata["card_path"], metadata["card_file"])
    right_card_file = gm.GameState.load_cards_from_file(metadata["card_path"], metadata["card_file"])
    
    for i in range(metadata["game_batch"]):
        left_cards = random.sample(left_card_file, gm.START_HANDS)
        right_cards = random.sample(right_card_file, gm.START_HANDS)
            
        new_game = gm.GameState(left_cards = left_cards, right_cards = right_cards)
        
        while(not new_game.is_end_of_game()):
            if new_game.current_player == gm.LEFT_PLAYER:
                # Record all the moves made by the learner
                (card, move) = player.get_action(new_game)
                states[i].append(fe.state2feature(new_game))
                (card_vector, move_vector) = player.action_to_vector(new_game, card, move)
                card_actions[i].append(np.expand_dims(card_vector, axis=0))
                move_actions[i].append(np.expand_dims(move_vector, axis=0))
                rewards[i].append(1)
            else:
                (card, move) = opponent.get_action(new_game)
            new_game.play_round(card, *move)
        
        rewards[i] = int(new_game.get_winner() == gm.LEFT_PLAYER) # I treat the loss and tie as the same
        
    return (states, card_actions, move_actions, rewards)

def train_on_batch(policy, states, card_actions, move_actions, rewards):
    for (state, card_action, move_action, result) in zip(states, card_actions, move_actions, rewards):
        policy.fit(np.concatenate(state, axis=0), 
                   np.concatenate(card_action, axis=0), 
                   np.concatenate(move_action, axis=0), 
                   result[0] == 1)

def train_on_result(policy, states, card_actions, move_actions, rewards):
    won_tuples = [(state, card, move, won) for (game_state, game_card, game_move, game_won) in zip(states, card_actions, move_actions, rewards) \
                  for (state, card, move, won) in zip(game_state, game_card, game_move, game_won) if won == 1]
    lost_tuples = [(state, card, move, won) for (game_state, game_card, game_move, game_won) in zip(states, card_actions, move_actions, rewards) \
                  for (state, card, move, won) in zip(game_state, game_card, game_move, game_won) if won == -1]
    if len(won_tuples) > 0: 
        won_tensors = np.transpose(won_tuples)
        policy.fit(np.concatenate(won_tensors[0], axis=0), 
                   np.concatenate(won_tensors[1], axis=0), 
                   np.concatenate(won_tensors[2], axis=0), 
                   1)
    
    if len(lost_tuples) > 0: 
        lost_tensors = np.transpose(lost_tuples)
        policy.fit(np.concatenate(lost_tensors[0], axis=0), 
                   np.concatenate(lost_tensors[1], axis=0), 
                   np.concatenate(lost_tensors[2], axis=0), 
                   0)
            
def run_training(cmd_line_args=None):
    import argparse
    parser = argparse.ArgumentParser(description='Train the policy network with Monte Carlo approach and exploring start')
    parser.add_argument("out_directory", help="Path to folder where the model params and metadata will be saved after each epoch.")
    parser.add_argument("--resume", help="Load latest weights in out_directory and resume", default=False, action="store_true")
    parser.add_argument("--model-json", help="JSON file for policy model in the output directory.", default = "model.json")
    parser.add_argument("--initial-weights", help="Path to HDF5 file with inital weights (i.e. result of supervised training).", default = ZEROTH_FILE)
    parser.add_argument("--learning-rate", help="Keras learning rate (Default: 0.01)", type=float, default=0.01)
    parser.add_argument("--save-every", help="Save policy as a new opponent every n batches (Default: 200)", type=int, default=200)
    parser.add_argument("--record-every", help="Save learner's weights every n batches (Default: 100)", type=int, default=100)
    parser.add_argument("--game-batch", help="Number of games per mini-batch (Default: 50)", type=int, default=50)
    parser.add_argument("--iterations", help="Number of training batches/iterations (Default: 50000)", type=int, default=5000)
    parser.add_argument("--card-path", help="The directory with the card set file (Default: {})".format(gm.DEFAULT_PATH), default=gm.DEFAULT_PATH)
    parser.add_argument("--card-file", help="The file containing the cards to play with (Default: {})".format(gm.DEFAULT_CARDS_FILE), default=gm.DEFAULT_CARDS_FILE)
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
    
    
    if not os.path.exists(os.path.join(args.card_path, args.card_file)):
        raise ValueError("Cannot resume without card file {} in the directory {}".format(args.card_file, args.card_path))
            
    
    metadata = {
            "out_directory": args.out_directory,
            "model_file": args.model_json,
            "init_weights": args.initial_weights,
            "learning_rate": args.learning_rate,
            "game_batch": args.game_batch,
            "save_every": args.save_every,
            "card_path": args.card_path,
            "card_file": args.card_file,
            "opponents": [ZEROTH_FILE],  # which weights from which to sample an opponent each batch
            "num_wins": {},  # number of wins for player in each batch
            "wins_per_opponent": {}
        }
                 
    if not args.resume:
        # starting the game from scratch
        player_weights = ZEROTH_FILE
        iter_start = 1
        player = NNPolicy(model_save_path = os.path.join(args.out_directory, args.model_json))
        Helper.save_metadata(metadata, args.out_directory, "metadata.json")
        player.save_model()
        # Create the Zeroth weight file
        player.model.save_weights(os.path.join(args.out_directory, player_weights))
    else:
        # Load the metadata
        if not os.path.exists(os.path.join(args.out_directory, "metadata.json")):
            raise ValueError("Cannot resume without metadata.json file in the output directory")
        with open(os.path.join(args.out_directory, "metadata.json"), "r") as f:
            old_metadata = json.load(f)
        
        # Merge the metadata in case any parameter changed
        metadata = {**old_metadata, **metadata}    
        
        # Load the model    
        if not os.path.exists(os.path.join(args.out_directory, args.model_json)):
            raise ValueError("Cannot resume without model json file in the output directory")
        args.model_json = os.path.join(args.out_directory, os.path.basename(args.model_json))
        
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
    
    game_pool = []
    LIMIT = 5000
    
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

        # Run games (and learn from results)
        (states, card_actions, move_actions, rewards) = simulate_games(player, opponent, metadata)
        
        game_pool = game_pool + list(zip((states, card_actions, move_actions, rewards)))
        
        if len(game_pool) > LIMIT:
            random.shuffle(game_pool)
            train_on_batch()
        #train_on_batch(player, states, card_actions, move_actions, rewards)
        games_won = sum(reward[0] == 1 for reward in rewards)
        games_lost = sum(reward[0] == -1 for reward in rewards)
        if args.verbose:
            print("In iteration {} winrate is {}, loserate is {} against opponent {}".format(i_iter,\
                                                                    round(games_won / metadata["game_batch"], 2), \
                                                                    round(games_lost / metadata["game_batch"], 2), \
                                                                    opp_weights))
            
        metadata["num_wins"][player_weights] = games_won
        if opp_weights in metadata["wins_per_opponent"]:
            metadata["wins_per_opponent"][opp_weights].append(games_won)
        else:
            metadata["wins_per_opponent"][opp_weights] = [games_won]

        # Save intermediate models.
        if i_iter % args.record_every == 0:
            player.model.save_weights(os.path.join(args.out_directory, player_weights))

        # Add player to batch of oppenents once in a while.
        if i_iter % args.save_every == 0:
            metadata["opponents"].append(player_weights)
        Helper.save_metadata(metadata, args.out_directory, "metadata.json")
        

        
        


if __name__ == '__main__':
    run_training()