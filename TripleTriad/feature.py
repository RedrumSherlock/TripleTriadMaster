import numpy as np

def get_board_numbers(state):
    return [3,4,5]

def get_opp_handcards(state):
    return [1, 2, 3, 4, 5]

def get_self_handcards(state):
    return [1,2,3,4,5]

def get_turn(state):
    return [state.turn]

VALID_FEATURES = {
    "board_numbers": {
        "size": 3,
        "function": get_board_numbers
    },
    "opp_cards": {
        "size": 5,
        "function": get_opp_handcards
    },
    "self_cards": {
        "size": 5,
        "function": get_self_handcards
    },
    "turn": {
        "size": 1,
        "function": get_turn
    }
}

def state2feature(state, feature_list):
    features = []
    feature_dim = 0
    for feature in feature_list:
        if feature in VALID_FEATURES:
            features = features + VALID_FEATURES[feature]["function"](state)
            feature_dim = feature_dim + VALID_FEATURES[feature]["size"]
    return feature_dim, features