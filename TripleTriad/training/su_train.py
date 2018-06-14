
"""
This module is designed to perform supervised training for the NN Policy to get close to our manually
crafted policy. Basically this is supposed to provide a decent initial weight for the reinforcement 
learning policy
"""


import TripleTriad.game as gm
from TripleTriad.player.NNPolicy import NNPolicy
from TripleTriad.player.basic_policy import BaselinePolicy
import TripleTriad.feature as fe
import TripleTriad.game_helper as Helper
from TripleTriad.training.mc_train import ZEROTH_FILE

import numpy as np
import os
import random

from keras.optimizers import SGD, Adam
from keras.callbacks import CSVLogger, EarlyStopping

def simulate_single_game(target_policy, new_game):
    states = []
    cards = []
    moves = []
    
    while(not new_game.is_end_of_game()):
        (card, move) = target_policy.get_action(new_game)
        states.append(fe.feature_without_batch_axis(new_game))
        (card_vector, move_vector) = target_policy.action_to_vector(new_game, card, move)
        cards.append(card_vector)
        moves.append(move_vector)
        new_game.play_round(card, *move)
        
    return states, cards, moves
    
    
def state_action_generator(target_policy, metadata):
    """
    Args:
        target_policy: a policy for the NNPolicy to learn to. We use the manually crafted policy here.
        metadata: a dictionary which contains the meta data for this training process
        
    Yields:
        states: a nparray with shape (n, dim, 10). Here n is batch_size*9 (total steps for each game is 9). dim is the dimension of all selected features for each card,
                and 10 is for each card in both hands.
        cards: a nparray with shape (n, 10). Here n is batch_size*9 (total steps for each game is 9). The second dimension is a one-hot vector specifying which card to pick.
        moves: a nparray with shape (n, 9). Here n is batch_size*9 (total steps for each game is 9). The second dimension is a one-hot vector specifying which position on the 
                board to play the card.
    """
    

    
    left_card_file = gm.GameState.load_cards_from_file(metadata["card_path"], metadata["card_file"])
    right_card_file = gm.GameState.load_cards_from_file(metadata["card_path"], metadata["card_file"])
        
    while True:
        all_states = []
        all_cards = []
        all_moves = []
        for idx in range(metadata["batch_size"]):
            left_cards = random.sample(left_card_file, gm.START_HANDS)
            right_cards = random.sample(right_card_file, gm.START_HANDS)
            new_game = gm.GameState(left_cards = left_cards, right_cards = right_cards)
            
            (states, cards, moves) = simulate_single_game(target_policy, new_game)
            all_states.append(states)
            all_cards.append(cards)
            all_moves.append(moves)
        np_states = np.array(all_states) # the shape should be steps_per_game x batch_size x feature_dim x 10. Would need to reshape to merge the first two dims 
        np_cards = np.array(all_cards)  # the shape should be steps_per_game x batch_size x 19. Would need to reshape to merge the first two dims
        np_moves = np.array(all_moves)  # the shape should be steps_per_game x batch_size x 19. Would need to reshape to merge the first two dims
        yield (np_states.reshape((-1,) + np_states.shape[2:]), \
            {"card_output": np.array(np_cards.reshape((-1,) + np_cards.shape[2:])), \
             "move_output": np.array(np_moves.reshape((-1,) + np_moves.shape[2:]))})
        del all_states
        del all_cards
        del all_moves


def run_training():
    import argparse
    parser = argparse.ArgumentParser(description='Train the policy network to simulate the baseline policy')
    parser.add_argument("out_directory", help="Path to folder where the model params and metadata will be saved after each epoch.")
    parser.add_argument("--initial-weights", help="Path to HDF5 file with inital weights.", default = ZEROTH_FILE)
    parser.add_argument("--model-json", help="JSON file for policy model in the output directory.", default = "model.json")
    parser.add_argument("--learning-rate", help="Keras learning rate (Default: 0.01)", type=float, default=0.01)
    parser.add_argument("--epoch", help="Number of epoches for training process (Default: 50)", type=int, default=50)
    parser.add_argument("--step-epoch", help="Number of step per epoch(Default: 1000)", type=int, default=1000)
    parser.add_argument("--batch-size", help="Number of games to simulate for each batch (Default: 50)", type=int, default=50)
    parser.add_argument("--val-steps", help="Number of steps for validation (Default: 1000)", type=int, default=1000)
    parser.add_argument("--result-file", help="The file to save results as csv )", default="result.csv")
    parser.add_argument("--card-path", help="The directory with the card set file (Default: {})".format(gm.DEFAULT_PATH), default=gm.DEFAULT_PATH)
    parser.add_argument("--card-file", help="The file containing the cards to play with (Default: {})".format(gm.DEFAULT_CARDS_FILE), default=gm.DEFAULT_CARDS_FILE)
    parser.add_argument("--verbose", "-v", help="Turn on verbose mode", default=True, action="store_true")
    
    args = parser.parse_args()

    if not os.path.exists(args.out_directory):
        if args.verbose:
            print("creating output directory {}".format(args.out_directory))
        os.makedirs(args.out_directory)
    
    
    if not os.path.exists(os.path.join(args.card_path, args.card_file)):
        raise ValueError("Cannot play the game without card file {} in the directory {}".format(args.card_file, args.card_path))
            
                
    metadata = {
        "out_directory": args.out_directory,
        "model_file": args.model_json,
        "init_weights": args.initial_weights,
        "learning_rate": args.learning_rate,
        "epoch": args.epoch,
        "step_epoch": args.step_epoch,
        "batch_size": args.batch_size,
        "val_steps": args.val_steps,
        "result_file": args.result_file,
        "card_path": args.card_path,
        "card_file": args.card_file
    }
    
    iter_start = 1
    player = NNPolicy(model_save_path = os.path.join(args.out_directory, args.model_json))
    Helper.save_metadata(metadata, args.out_directory, "su_metadata.json")
    player.save_model()

    target = BaselinePolicy()
    
    optimizer = SGD(lr=metadata["learning_rate"])
    player.model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    
    train_generator = state_action_generator(target, metadata)
    validation_generator = state_action_generator(target, metadata)
    
    csv_logger = CSVLogger(os.path.join(args.out_directory, args.result_file), append=True)
    stopper = EarlyStopping(monitor='loss', patience=3)

    player.model.fit_generator(
        generator=train_generator,
        steps_per_epoch=metadata["step_epoch"],
        epochs=metadata["epoch"],
        callbacks=[csv_logger, stopper],
        validation_data=validation_generator,
        validation_steps=metadata["val_steps"])
    
    player.model.save_weights(os.path.join(args.out_directory, ZEROTH_FILE))

if __name__ == '__main__':
    run_training()