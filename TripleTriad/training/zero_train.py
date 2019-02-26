from TripleTriad.player.ZeroNN import ZeroPolicy
from TripleTriad.run import MAX_POOL_SIZE

import numpy as np
import random

from keras.optimizers import Adam

# The size of batch for each training
TRAIN_BATCH_SIZE = 2048
# The number of epoch before it can be evaluated
TRAN_EPOCH = 1000
# Learning rate
LEARNING_RATE = 0.01


def run_training(queue, train_manager):

    player = ZeroPolicy(train_manager.out_dir)
    player.save_model()
    optimizer = Adam(lr=LEARNING_RATE)
    player.model.compile(loss=log_loss, optimizer=optimizer)

    game_pool = []

    while True:
        # Before training update the game pool
        if queue.qsize() + len(game_pool) > MAX_POOL_SIZE:
            pool_size = 0
            while pool_size < MAX_POOL_SIZE and not queue.empty():
                game_pool.append(queue.get())
                pool_size = pool_size + 1

        # fetch the batch from game pool
        game_batch = []
        for i in range(TRAIN_BATCH_SIZE):
            # pick a random position from a random game in the pool
            position = random.choice(random.choice(game_pool))
            game_batch.append(position)

        train_on_batch(player)
        train_manager.evaluate_policy(player, game_batch)


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


def train_on_batch(policy, states, card_actions, move_actions, rewards):
    for (state, card_action, move_action, result) in zip(states, card_actions, move_actions, rewards):
        policy.fit(np.concatenate(state, axis=0),
                   np.concatenate(card_action, axis=0),
                   np.concatenate(move_action, axis=0),
                   result[0] == 1)


def train_on_result(policy, states, card_actions, move_actions, rewards):
    won_tuples = [(state, card, move, won) for (game_state, game_card, game_move, game_won) in
                  zip(states, card_actions, move_actions, rewards) \
                  for (state, card, move, won) in zip(game_state, game_card, game_move, game_won) if won == 1]
    lost_tuples = [(state, card, move, won) for (game_state, game_card, game_move, game_won) in
                   zip(states, card_actions, move_actions, rewards) \
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
