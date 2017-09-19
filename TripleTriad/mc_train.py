import os
import json
import re
import numpy as np
from shutil import copyfile
from keras.optimizers import SGD
import keras.backend as K
from TripleTriad.policy import NNPolicy

def log_loss(y_true, y_pred):
    '''Keras 'loss' function for the REINFORCE algorithm, where y_true is the action that was
    taken, and updates with the negative gradient will make that action more likely. We use the
    negative gradient because keras expects training data to minimize a loss function.
    '''
    return -y_true * K.log(K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon()))

def run_training(cmd_line_args=None):
    import argparse
    parser = argparse.ArgumentParser(description='Train the policy network with Monte Carlo approach and exploring start')
    parser.add_argument("out_directory", help="Path to folder where the model params and metadata will be saved after each epoch.")
    parser.add_argument("--resume", help="Load latest weights in out_directory and resume", default=False, action="store_true")
    parser.add_argument("--model_json", help="Path to policy model JSON.", default = "")
    parser.add_argument("--initial_weights", help="Path to HDF5 file with inital weights (i.e. result of supervised training).", default = "")
    parser.add_argument("--learning-rate", help="Keras learning rate (Default: 0.001)", type=float, default=0.001)
    parser.add_argument("--save-every", help="Save policy as a new opponent every n batches (Default: 500)", type=int, default=500)
    parser.add_argument("--record-every", help="Save learner's weights every n batches (Default: 1)", type=int, default=1)
    parser.add_argument("--game-batch", help="Number of games per mini-batch (Default: 20)", type=int, default=20)
    parser.add_argument("--iterations", help="Number of training batches/iterations (Default: 10000)", type=int, default=10000)
    parser.add_argument("--verbose", "-v", help="Turn on verbose mode", default=True, action="store_true")
    
    # Baseline function (TODO) default lambda state: 0  (receives either file
    # paths to JSON and weights or None, in which case it uses default baseline 0)
    if cmd_line_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(cmd_line_args)

    ZEROTH_FILE = "weights.00000.hdf5"

    if args.resume:
        if not os.path.exists(os.path.join(args.out_directory, "metadata.json")):
            raise ValueError("Cannot resume without existing output directory")

    if not os.path.exists(args.out_directory):
        if args.verbose:
            print("creating output directory {}".format(args.out_directory))
        os.makedirs(args.out_directory)

    if not args.resume:
        # starting the game from scratch
        player_weights = ZEROTH_FILE
        iter_start = 1
        player = NNPolicy()
    else:
        # if resuming, we expect initial_weights to be just a
        # "weights.#####.hdf5" file, not a full path
        if not re.match(r"weights\.\d{5}\.hdf5", args.initial_weights):
            raise ValueError("Expected to resume from weights file with name 'weights.#####.hdf5'")
        args.initial_weights = os.path.join(args.out_directory,
                                            os.path.basename(args.initial_weights))
        if not os.path.exists(args.initial_weights):
            raise ValueError("Cannot resume; weights {} do not exist".format(args.initial_weights))
        elif args.verbose:
            print("Resuming with weights {}".format(args.initial_weights))
        player_weights = os.path.basename(args.initial_weights)
        iter_start = 1 + int(player_weights[8:13])
        player = NNPolicy(model_load_path = args.model_json)
        policy.model.load_weights(args.initial_weights)

    opponent = player.clone()

    if args.verbose:
        print("created player and opponent")

    if not args.resume:
        metadata = {
            "model_file": args.model_json,
            "init_weights": args.initial_weights,
            "learning_rate": args.learning_rate,
            "temperature": args.policy_temp,
            "game_batch": args.game_batch,
            "opponents": [ZEROTH_FILE],  # which weights from which to sample an opponent each batch
            "win_ratio": {}  # map from player to tuple of (opponent, win ratio) Useful for
                             # validating in lieu of 'accuracy/loss'
        }
    else:
        with open(os.path.join(args.out_directory, "metadata.json"), "r") as f:
            metadata = json.load(f)

    # Append args of current run to history of full command args.
    metadata["cmd_line_args"] = metadata.get("cmd_line_args", [])
    metadata["cmd_line_args"].append(vars(args))

    def save_metadata():
        with open(os.path.join(args.out_directory, "metadata.json"), "w") as f:
            json.dump(metadata, f, sort_keys=True, indent=2)

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
        opponent.policy.model.load_weights(opp_path)
        if args.verbose:
            print("Batch {}\tsampled opponent is {}".format(i_iter, opp_weights))

        # Run games (and learn from results). Keep track of the win ratio vs each opponent over
        # time.
        win_ratio = run_n_games(optimizer, player, opponent, args.game_batch)
        metadata["win_ratio"][player_weights] = (opp_weights, win_ratio)

        # Save intermediate models.
        if i_iter % args.record_every == 0:
            player.policy.model.save_weights(os.path.join(args.out_directory, player_weights))

        # Add player to batch of oppenents once in a while.
        if i_iter % args.save_every == 0:
            metadata["opponents"].append(player_weights)
        save_metadata()


if __name__ == '__main__':
    run_training()