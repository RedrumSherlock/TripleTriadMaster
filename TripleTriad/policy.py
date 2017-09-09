import numpy as np

DEFAULT_FEATURES = [
    "board_numbers",
    "opp_handcards",
    "self_handcards",
    "turn"
    ]

class RandomPolicy():
    # This is an equiprobable policy that simply randomly pick one move from all the legal moves
        
    def get_action(self, state):
        moves = state.get_legal_moves()
        return np.random.choice(moves)
        