import random
import TripleTriad.game as gm
import TripleTriad.game_helper as Helper


class Policy():
    # This is an the base class for all policies
    
    def __init__(self):
        pass
        
    def clone(self):
        raise NotImplementedError("clone not Implemented!")
        
    def get_action(self, state):
        raise NotImplementedError("get_action not Implemented!")
    
    def action_to_vector(self, state, card, move):
        card_idx = [0] * (2 * gm.START_HANDS)
        board_idx = [0] * (gm.BOARD_SIZE ** 2)
        for i in range(2 * gm.START_HANDS):
            if i < gm.START_HANDS and state.left_cards[i] is card:
                card_idx[i] = 1
            if i >= gm.START_HANDS and state.right_cards[i - gm.START_HANDS] is card:
                card_idx[i] = 1
        board_idx[Helper.tuple2idx(gm.BOARD_SIZE, *move)] = 1
        return card_idx, board_idx

class RandomPolicy(Policy):
    # This is an equiprobable policy that simply randomly pick one move from all the legal moves
        
    def get_action(self, state):
        move = random.choice(state.get_legal_moves())
        card = random.choice(state.get_unplayed_cards())
        return card, move
