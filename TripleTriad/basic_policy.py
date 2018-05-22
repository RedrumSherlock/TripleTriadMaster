import TripleTriad.game_helper as Helper
import TripleTriad.game as Game
from TripleTriad.policy import Policy
import numpy as np

class BasicPolicy(Policy):
    """
    This is a manual compiled basic policy rule of how to play this game. We simply choose the card from available hand cards that is 
    able to flip the most of opponent cards on the board. If there are multiple choices, we choose the one with the least value. This 
    is expected to be much better than the random policy, but still far from the optimal play 
    """
    
    def __init__(self):
        pass
        
    def get_action(self, state):
        
        unplayed_cards = state.get_unplayed_cards()
        flips = [0] * len(unplayed_cards)    # How many flips can this card make at most 
        moves = [0] * len(unplayed_cards)    # At which position this card plays can make the most flips
        values = [0] * len(unplayed_cards)    # The value of this card
        for card in unplayed_cards:
            values[i] = card.get_value()
            for move in state.get_legal_moves():
                cards_flipped = state.flip_cards(card, move[0], move[1], False)
                if cards_flipped > flips[i]:
                    flips[i] = cards_flipped
                    play_at[i] = move
        
        if sum(flips) == 0:
            pass
        else:
            max_flip = max(flips)     
            card_index = values.index(min(values[i] for i,v in enumerate(flips) if v == max_flip))
            move = moves[card_index]
                                  
        return unplayed_cards(card_index), move
