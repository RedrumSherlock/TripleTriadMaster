import TripleTriad.game_helper as Helper
import TripleTriad.game as Game
from TripleTriad.player.policy import Policy
import numpy as np
import random
from tensorflow.contrib.graph_editor.util import is_iterable

class BasicPolicy(Policy):
    """
    This is a manual compiled basic policy rule of how to play this game. We simply choose the card from available hand cards that is 
    able to flip the most of opponent cards on the board. If there are multiple choices, we choose the one with the least value. This 
    is expected to be much better than the random policy, but still far from the optimal play 
    
    Based on a rough estimate this policy can achieve 78% win rate against random policy, losing only 7% of the games
    """
    
    def __init__(self):
        pass
        
    def get_action(self, state):
        
        unplayed_cards = state.get_unplayed_cards()
        legal_moves = state.get_legal_moves()
        flips = [0] * len(unplayed_cards)    # How many flips can this card make at most 
        moves = [legal_moves[0]] * len(unplayed_cards)    # Which position this card plays at can make the most flips
        values = [0] * len(unplayed_cards)    # The value of this card
        
        for i, card in enumerate(unplayed_cards):
            values[i] = card.get_value()
            for move in legal_moves:
                cards_flipped = state.flip_cards(card, move[0], move[1], False)
                if cards_flipped > flips[i]:
                    flips[i] = cards_flipped
                    moves[i] = move
        
        if sum(flips) == 0:
            # This means nothing will be flipped no matter what do we play. In this case we simply pick the card with the least value, and play it in 
            # a random available spot following the order: corner > side > middle
            card_index = np.argmin(values)
            corners = [move for move in legal_moves if Helper.is_corner(*move)]
            if len(corners) > 0:
                move = random.choice(corners)
            else:
                sides = [move for move in legal_moves if Helper.is_side(*move)]
                if len(sides) > 0:
                    move = random.choice(sides)
                else:
                    move = (1, 1)
            
        else:
            # We pick the card that is able to make the most flips, but with the lowest value
            max_flip = max(flips)     
            card_index = values.index(min(values[i] for i,v in enumerate(flips) if v == max_flip))
            move = moves[card_index]
                                  
        return unplayed_cards[card_index], move
