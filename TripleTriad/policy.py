import TripleTriad.game_helper as Helper
import TripleTriad.game as Game
import numpy as np
import random


class Policy():
    # This is an the base class for all policies
    
    def __init__(self):
        pass
        
    def clone(self):
        raise NotImplementedError("clone not Implemented!")
        
    def get_action(self, state):
        raise NotImplementedError("get_action not Implemented!")
    
    

class RandomPolicy(Policy):
    # This is an equiprobable policy that simply randomly pick one move from all the legal moves
        
    def get_action(self, state):
        move = random.choice(state.get_legal_moves())
        card = random.choice(state.get_unplayed_cards())
        return card, move
