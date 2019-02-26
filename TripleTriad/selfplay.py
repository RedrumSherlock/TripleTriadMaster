"""
Created on Dec 13, 2018

@author: Mengliao Wang

This is the main program to play the best policy against itself to execute the games for training

"""

from TripleTriad.game import GameState, START_HANDS, LEFT_PLAYER
from TripleTriad.feature import feature_without_batch_axis
from TripleTriad.run import SELF_PLAY_GAMES

import random


def self_play(queue, train_manager):
    # Load the best player available
    player = train_manager.get_best_policy()
    opponent = player.clone()

    # Prepare the game state
    left_card_file = GameState.load_cards_from_file(train_manager.card_path, train_manager.card_file)
    right_card_file = GameState.load_cards_from_file(train_manager.card_path, train_manager.card_file)

    while True:
        left_cards = random.sample(left_card_file, START_HANDS)
        right_cards = random.sample(right_card_file, START_HANDS)
        new_game = GameState(left_cards=left_cards, right_cards=right_cards)
        game_positions = simulate_single_game(player, opponent, new_game)
        queue.put(game_positions)

        # If played enough games, check for new best player
        if queue.qsize() > SELF_PLAY_GAMES:
            player = train_manager.get_best_policy()
            opponent = player.clone()


def simulate_single_game(player, opponent, game):
    states = []
    actions = []
    results = []

    while not game.is_end_of_game():
        # Player is always on the left, and the opponent is always on the right. Randomly picks who starts the game.
        if game.current_player == LEFT_PLAYER:
            (card, move) = player.get_action(game)
        else:
            (card, move) = opponent.get_action(game)
        states.append(feature_without_batch_axis(game))
        actions.append(player.get_MCTS_probs())
        results.append(1)
        game.play_round(card, *move)

    return (states, actions, results)