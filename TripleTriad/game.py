'''
Created on Aug 30, 2017

@author: Mengliao Wang
'''
import numpy as np
import warnings


# Left-hand side player and Right-hand side player 
LEFT_PLAYER = -1
TIE = 0
RIGHT_PLAYER = 1

# the number of hards for each player at the beginning of the game, and the size
# of the board. These should not be changed.
START_HANDS = 5
BOARD_SIZE = 3

# Check the detail of rules at http://ffxivtriad.com/rules. Right now only the all open rule is implemented
rule_list = [
    "all_open",
    "three_open",
    "sudden_death",
    "random",
    "order",
    "chaos",
    "reverse",
    "fallen_ace",
    "same",
    "combo",
    "plus",
    "ascension",
    "descension",
    "swap"]

class GameState(object):
    '''
    Here are the setting of the game:
    rule
    '''


    def __init__(self, left_cards=[], right_cards=[], current_player=np.random.choice([LEFT_PLAYER, RIGHT_PLAYER]), rules=['all_open']):
        self.rules = rules
        self.current_player = current_player
        self.left_cards = left_cards
        self.right_cards = right_cards
        
        # Initialize the board
        self.board = [None] * (BOARD_SIZE * BOARD_SIZE)
        
        if len(left_cards) == 0:
            load_cards(self.left_cards)
        
        if len(right_cards) == 0:
            load_cards(self.right_cards)
            
        # Check if all the rules are valid
        for rule in self.rules:
            if not rule in rule_list:
                warnings.warn("Rule %s is not valid. Ignoring this rule" % rule) 
        
        # Apply the rules here
        if "all_open" in self.rules:
            for card in self.left_cards:
                card.visible = True
            for card in self.right_cards:
                card.visible = True
    
    def on_Board(self, x_pos, y_pos):
        # The board is a 3x3 matrix, and the index ranges from 0 to 2 for both x(horizontal) and y(vertical)
         return x_pos >= 0 and x_pos <= 2 and y_pos >= 0 and y_pos <= 2
     
    def get_card(self, x_pos, y_pos):
        # returns the card on the board. If out side of the board returns None
        if on_Board(x_pos, y_pos):
            return self.board[x_pos + y_pos * BOARD_SIZE]
        else:
            return None
    
    def place_card(self, x_pos, y_pos, card):
        # Drop the card on the board based on the coordinates
        if on_Board(x_pos, y_pos):
            self.board[x_pos + y_pos * BOARD_SIZE] = card
        
    def get_neighbours(self, x_pos, y_pos):
        # Returns a list of cards placed on board for the x_pos, y_pos. The list
        # will always have 4 elements, following clockwise [top, right, bottom,
        # left]
        neighbours = []
        neighbours.append(get_card(x_pos, y_pos - 1))
        neighbours.append(get_card(x_pos + 1, y_pos))
        neighbours.append(get_card(x_pos, y_pos + 1))
        neighbours.append(get_card(x_pos - 1, y_pos))
        return neighbours
    
    
    def flip_cards(self, card, neighbours):
        # Apply the basic rule here. Will be tuned for special rules such as same or combo.
        # Flip top, right, bot, left
        for i in range(4):
            if neighbours[i] is not None and neighbours[i].owner != self.current_player:
                # Compare the top edge
                if i == 0 and card.get_top() > neighbours[i].get_bottom():
                    neighbours[i].owner = self.current_player
                # Compare the right edge
                if i == 1 and card.get_right() > neighbours[i].get_left():
                    neighbours[i].owner = self.current_player
                # Compare the bottom edge
                if i == 2 and card.get_bottom() > neighbours[i].get_top():
                    neighbours[i].owner = self.current_player
                # Compare the left edge
                if i == 3 and card.get_left() > neighbours[i].get_right():
                    neighbours[i].owner = self.current_player
                    
    def get_winner(self):
        if len(filter(lambda x: x is None, self.board)) == 0:
            left_cards = len(filter(lambda l: l is not None and l.owner == LEFT_PLAYER, self.board))
            right_cards = len(filter(lambda r: r is not None and r.owner == RIGHT_PLAYER, self.board))
            if left_cards == right_cards:
                return TIE
            elif left_cards > right_cards:
                return LEFT_PLAYER
            else:
                return RIGHT_PLAYER
        else:
            return None
        
    # Play a card at position [x_pos, y_pos]            
    def play_round(self, x_pos, y_pos, card):
        card.owner = self.current_player
        place_card(x_pos, y_pos, card)
        neighbours = et_neighbours(x_pos, y_pos)
        
        flip_cards(card, neighbours)
        self.current_player = -1 * player
        
    
    # Display the status of the board
    def print_board(self):
        for i in range(BOARD_SIZE):
            print("-------------------------------")
            top_line = "|"
            mid_line = "|"
            bot_line = "|"
            for j in range(BOARD_SIZE):
                card = get_card(i, j)
                top_line = top_line + "   " if card is None else ( " " + str(card.get_top()) + " |")
                mid_line = mid_line + "   " if card is None else (str(card.get_left()) + " " + str(card.get_right()) + "|")
                bot_line = top_line + "   " if card is None else ( " " + str(card.get_bottom()) + " |")
            print(top_line)
            print(mid_line)
            print(bot_line)
        print("-------------------------------")
        
class Card(object):

    '''
    The properties of a card:
    Number: The numbers of a card shown on the top left corner. It is a 4-elements list on a clockwise [top, right, bottom, left]. Number ranges from 1 to 10 (10 is ace)
    Owner: the player that owns this card
    Visible: whether the opponent can see this card
    On_hand: Whether this card is still in player's hand or is on the board
    Rank: the rank of this card. If -1 it means this card is not classified
    Element: Only for special rules. Not implemented yet.
    '''
    
    def __init__(self, numbers, owner, visible = False, on_hand = True, rank=-1,  element = -1):
        self.numbers = numbers
        self.owner = owner
        self.visible = visible
        self.on_hand = on_hand
        self.rank = rank
        self.element = element
        
    def get_top(self):
        return self.numbers[0]
    
    def get_right(self):
        return self.numbers[1]
    
    def get_bottom(self):
        return self.numbers[2]
    
    def get_left(self):
        return self.numbers[3]
    
    
    