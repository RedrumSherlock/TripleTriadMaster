import TripleTriad.game as Game
import numpy as np


def ActionToCardMove(state, action):
    # action is a one_hot vector for the index of the card picked from state.left_cards + state.right_cards, 
    # and move is one of the 9 cells on the board, left to right and top to bottom. 
    # So it looks like this: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0,    0, 0, 0, 1, 0, 0, 0, 0, 0]
    # This example means we will pick the 5th card from the left player's hands, and place on the (0, 1) cell on the board
    # It will return the (card, move) pair, where card is the Card object, and move is the (x, y) tuple
    if len(action) != 2 * Game.START_HANDS + Game.BOARD_SIZE * Game.BOARD_SIZE:
        raise ValueError("The action must have 19 dimensions")
    if np.sum(action[:2 * Game.START_HANDS]) != 1 or np.sum(action[2 * Game.START_HANDS:]) != 1:
        raise ValueError("The action must be made of two one-hot vectors for card and move respectively")
    card_index = np.argmax(action[:2 * Game.START_HANDS], axis = 0)
    board_index = np.argmax(action[2 * Game.START_HANDS:], axis = 0)

    return ( (state.left_cards + state.right_cards)[card_index], (board_index % Game.BOARD_SIZE, int(np.floor(board_index/Game.BOARD_SIZE))) )