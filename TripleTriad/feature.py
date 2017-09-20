import numpy as np
import TripleTriad.game as Game

def get_number(state, index):
    result = np.zeros((1, 2 * Game.START_HANDS))
    for i in range(2 * Game.START_HANDS):
        if i < Game.START_HANDS:
            card = state.left_cards[i]
        else:
            card = state.right_cards[i - Game.START_HANDS]
        
        if not card.visible and card.owner != state.current_player:
            result[0, i] = 0
        else:
            result[0, i] = card.get_number(index)
    return result

def get_number_one_hot(state, index):
    result = np.zeros((11, 2 * Game.START_HANDS))
    for i in range(2 * Game.START_HANDS):
        if i < Game.START_HANDS:
            card = state.left_cards[i]
        else:
            card = state.right_cards[i - Game.START_HANDS]
        
        if not card.visible and card.owner != state.current_player:
            result[0, i] = 1
        else:
            result[card.get_number(index), i] = 1
    return result


def get_top_number(state):
    return get_number(state, Game.TOP_INDEX)

def get_top_number_one_hot(state):
    return get_number_one_hot(state, Game.TOP_INDEX)
    
def get_right_number(state):
    return get_number(state, Game.RIGHT_INDEX)

def get_right_number_one_hot(state):
    return get_number_one_hot(state, Game.RIGHT_INDEX)

def get_bottom_number(state):
    return get_number(state, Game.BOTTOM_INDEX)

def get_bottom_number_one_hot(state):
    return get_number_one_hot(state, Game.BOTTOM_INDEX)

def get_left_number(state):
    return get_number(state, Game.LEFT_INDEX)

def get_left_number_one_hot(state):
    return get_number_one_hot(state, Game.LEFT_INDEX)


def get_position(state):
    result = np.zeros((11, 2 * Game.START_HANDS))
    for i in range(2 * Game.START_HANDS):
        if i < Game.START_HANDS:
            card = state.left_cards[i]
        else:
            card = state.right_cards[i - Game.START_HANDS]
        
        if state.on_Board(*card.position):
            result[card.position[0] + card.position[1] * Game.BOARD_SIZE, i] = 1
        elif card.owner == Game.LEFT_PLAYER:
            result[9, i] = 1
        elif card.owner == Game.RIGHT_PLAYER:
            result[10, i] = 1
    return result    

def get_owner(state):
    result = np.zeros((1, 2 * Game.START_HANDS))
    for i in range(2 * Game.START_HANDS):
        if i < Game.START_HANDS:
            card = state.left_cards[i]
        else:
            card = state.right_cards[i - Game.START_HANDS]
        
        result[0, i] = (card.owner == Game.LEFT_PLAYER)
    return result    

def get_player(state):
    return np.ones((1, 2 * Game.START_HANDS)) * (state.current_player == Game.LEFT_PLAYER)

def get_rank(state):
    result = np.zeros((1, 2 * Game.START_HANDS))
    for i in range(2 * Game.START_HANDS):
        if i < Game.START_HANDS:
            card = state.left_cards[i]
        else:
            card = state.right_cards[i - Game.START_HANDS]
        
        if not card.visible and card.owner != state.current_player:
            result[0, i] = 0
        else:
            result[0, i] = card.rank
    return result    

def get_rank_one_hot(state):
    result = np.zeros((Game.MAX_RANK_LEVEL + 1, 2 * Game.START_HANDS))
    for i in range(2 * Game.START_HANDS):
        if i < Game.START_HANDS:
            card = state.left_cards[i]
        else:
            card = state.right_cards[i - Game.START_HANDS]
        
        if (not card.visible and card.owner != state.current_player) or card.get_rank() < 1:
            result[0, i] = 1
        else:
            result[card.get_rank(), i] = 1
    return result    

def get_element(state):
    # TODO - To be implemented. This feature is not being used right now
    result = np.zeros((4, 2 * Game.START_HANDS))
    return result
    
def get_turn(state):
    # TODO - To be implemented. This feature is not being used right now
    result = np.zeros((9, 2 * Game.START_HANDS))
    return result

VALID_FEATURES = {
    # One-hot encoding or integer for the card numbers? I think integer makes sense but one-hot might be better
    "top_number": {
        "size": 1,
        "function": get_top_number
    },
    "right_number": {
        "size": 1,
        "function": get_right_number
    },
    "bottom_number": {
        "size": 1,
        "function": get_bottom_number
    },
    "left_number": {
        "size": 1,
        "function": get_left_number
    }, 
    # The one-hot encoding for the numbers. Number could vary from 1 to 10(i.e. Ace). 
    # But another case is when the card is invisible then the number is unknown so additional dimension for this
    "top_number_one_hot": {
        "size": 11,
        "function": get_top_number_one_hot
    },
    "right_number_one_hot": {
        "size": 11,
        "function": get_right_number_one_hot
    },
    "bottom_number_one_hot": {
        "size": 11,
        "function": get_bottom_number_one_hot
    },
    "left_number_one_hot": {
        "size": 11,
        "function": get_left_number_one_hot
    },    
    # The position can be one of the 9 cells on the board, or in the hand of either player                                                      
    "position": {
        "size": 11,
        "function": get_position
    },
    # Whether the owner of the card is LEFT_PLAYER
    "owner": {
        "size": 1,
        "function": get_owner
    },        
    # Similar to the card numbers: integer makes sense but one-hot might be better      
    "rank": {
        "size": 1,
        "function": get_rank
    },
    "rank_one_hot": {
        "size": Game.MAX_RANK_LEVEL + 1,
        "function": get_rank_one_hot
    },
    "element": {
        "size": 4,
        "function": get_element
    },
    # How many turns this card has been placed on the board. Don't think this helps on the gameplay, but will leave it here 
    "turn": {
        "size": 9,
        "function": get_turn
    },
    # Whether the player is LEFT_PLAYER
    "player": {
        "size": 1,
        "function": get_player
    }
}

DEFAULT_FEATURES = [
    "top_number", "right_number", "bottom_number", "left_number", "position",
    "owner", "player"] # Total dimension is 17 without one-hot encoding on the card numbers

def get_feature_dim(feature_list = DEFAULT_FEATURES):
    feature_dim = 0
    for feature in feature_list:
        if feature in VALID_FEATURES:
            feature_dim = feature_dim + VALID_FEATURES[feature]["size"]
        else:
            raise ValueError("Unknown feature: %s" % feature)
    return feature_dim

def state2feature(state, feature_list=DEFAULT_FEATURES):
    features = []
    for feature in feature_list:
        if feature in VALID_FEATURES:
            features.append(VALID_FEATURES[feature]["function"](state))
        else:
            raise ValueError("Unknown feature: %s" % feature)
    return np.concatenate(features, axis = 0)