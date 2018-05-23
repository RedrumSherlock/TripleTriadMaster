'''
Created on Aug 30, 2017

@author: Mengliao Wang
'''
import numpy as np
import warnings
import random
import os
import csv
import TripleTriad.game_helper as Helper


# Left-hand side player and Right-hand side player 
LEFT_PLAYER = 1
NO_ONE = 0
RIGHT_PLAYER = -1

# the number of hards for each player at the beginning of the game, and the size
# of the board. These should not be changed.
START_HANDS = 5
BOARD_SIZE = 3


# the default cardset for both players
DEFAULT_PATH = "test_cards"
DEFAULT_CARDS_FILE = "cards.csv"

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

TOP_INDEX = 0
RIGHT_INDEX = 1
BOTTOM_INDEX = 2
LEFT_INDEX = 3

# The maximum rank level of a card
MAX_RANK_LEVEL = 5

class GameState(object):


    def __init__(self, left_cards=None, right_cards=None, path=DEFAULT_PATH,
                 left_file=DEFAULT_CARDS_FILE, right_file=DEFAULT_CARDS_FILE,
                 current_player=None, rules=None):
        if rules is None:
            self.rules = ['all_open']
        if current_player is None:
            self.current_player = np.random.choice([LEFT_PLAYER, RIGHT_PLAYER])
        self.left_cards = left_cards
        self.right_cards = right_cards
        self.path = path
        self.left_file = left_file
        self.right_file = right_file
        self.turn = 0
        self.board_size = BOARD_SIZE
        self.start_hands = START_HANDS
        
        # Initialize the board
        self.board = [None] * (BOARD_SIZE ** 2)
        
        # If no cards were given, randomly choose 5 from the card set. Also set their owner to be left player
        if self.left_cards is None or len(self.left_cards) == 0:
            self.left_cards = random.sample(load_cards_from_file(self.path, self.left_file), START_HANDS)
        elif len(self.left_cards) != START_HANDS:
            self.left_cards = random.sample(self.left_cards, START_HANDS)
        for card in self.left_cards:
            card.reset()
            card.owner = LEFT_PLAYER
        
        # Same as above but for right player
        if self.right_cards is None or len(self.right_cards) == 0:
            self.right_cards = random.sample(load_cards_from_file(self.path, self.right_file), START_HANDS)
        elif len(self.right_cards) != START_HANDS:
            self.right_cards = random.sample(self.right_cards, START_HANDS)
        for card in self.right_cards:
            card.reset()
            card.owner = RIGHT_PLAYER
            
        # Check if all the rules are valid
        for rule in self.rules:
            if rule not in rule_list:
                warnings.warn("Rule %s is not valid. Ignoring this rule" % rule) 
        
        # Apply the rules here
        if "all_open" in self.rules:
            for card in self.left_cards:
                card.visible = True
            for card in self.right_cards:
                card.visible = True
    
    def on_Board(self, x_pos, y_pos):
        # The board is a 3x3 matrix, and the index ranges from 0 to 2 for both x(horizontal, columns) and y(vertical, rows)
         return x_pos >= 0 and x_pos < BOARD_SIZE and y_pos >= 0 and y_pos < BOARD_SIZE
     
    def get_card(self, x_pos, y_pos):
        # returns the card on the board. If out side of the board returns None
        if self.on_Board(x_pos, y_pos):
            return self.board[Helper.tuple2idx(BOARD_SIZE, x_pos, y_pos)]
        else:
            return None
    
    def get_handcard(self, card_index):
        # returns the card on the board. If out side of the board returns None
        return (self.left_cards + self.right_cards)[card_index]
    
    def place_card(self, card, x_pos, y_pos):
        # Drop the card on the board based on the coordinates
        if self.on_Board(x_pos, y_pos):
            self.board[Helper.tuple2idx(BOARD_SIZE, x_pos, y_pos)] = card
            card.position = (x_pos, y_pos)
            card.visible = True
        
    def get_neighbours(self, x_pos, y_pos):
        # Returns a list of cards placed on board for the x_pos, y_pos. The list
        # will always have 4 elements, following clockwise [top, right, bottom,
        # left]
        neighbours = []
        neighbours.append(self.get_card(x_pos, y_pos - 1))
        neighbours.append(self.get_card(x_pos + 1, y_pos))
        neighbours.append(self.get_card(x_pos, y_pos + 1))
        neighbours.append(self.get_card(x_pos - 1, y_pos))
        return neighbours
    
    
    def flip_cards(self, card, x_pos, y_pos, commit=True):
        # Apply the basic rule here. Will be tuned for special rules such as same or combo.
        # Flip top, right, bot, left
        flipped = 0
        neighbours = self.get_neighbours(x_pos, y_pos)
        for i in range(4):
            if neighbours[i] is not None and neighbours[i].owner != self.current_player:
                # Compare the top edge
                if i == 0 and card.get_top() > neighbours[i].get_bottom():
                    flipped += 1
                    if commit: neighbours[i].owner = self.current_player
                # Compare the right edge
                if i == 1 and card.get_right() > neighbours[i].get_left():
                    flipped += 1
                    if commit:  neighbours[i].owner = self.current_player
                # Compare the bottom edge
                if i == 2 and card.get_bottom() > neighbours[i].get_top():
                    flipped += 1
                    if commit: neighbours[i].owner = self.current_player
                # Compare the left edge
                if i == 3 and card.get_left() > neighbours[i].get_right():
                    flipped += 1
                    if commit: neighbours[i].owner = self.current_player
        return flipped
                
    def is_end_of_game(self):
        return sum(1 for _ in filter(lambda x: x is None, self.board)) == 0
    
    def get_winner(self):
        if self.is_end_of_game():
            left_cards = sum(1 for _ in filter(lambda l: l.owner == LEFT_PLAYER, self.left_cards + self.right_cards))
            right_cards = START_HANDS * 2 - left_cards
            if left_cards == right_cards:
                return NO_ONE
            elif left_cards > right_cards:
                return LEFT_PLAYER
            else:
                return RIGHT_PLAYER
        else:
            return None
    
    def get_legal_moves(self):
        moves = []
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                if self.board[Helper.tuple2idx(BOARD_SIZE, x, y)] is None:
                    moves.append((x, y))
        return moves
    
    def get_legal_moves_by_index(self):
        moves = [0] * (BOARD_SIZE ** 2)
        for i in range(len(self.board)):
            moves[i] = (self.board[i] is None)
        return moves
    
    def _is_card_playable(self, card):
        return (not self.on_Board(*card.position)) and card.owner == self.current_player
        
    def get_unplayed_cards(self):
        return list(filter(lambda card: self._is_card_playable(card), self.left_cards + self.right_cards))
    
    def get_unplayed_cards_by_index(self):
        cards = [0] * (2 * START_HANDS)
        all_cards = self.left_cards + self.right_cards
        for i in range(len(all_cards)):
            cards[i] = self._is_card_playable(all_cards[i]) * 1
        return cards
       
    # Play a card at position [x_pos, y_pos]            
    def play_round(self, card, x_pos, y_pos):
        card.owner = self.current_player
        self.place_card(card, x_pos, y_pos)
        
        self.flip_cards(card, x_pos, y_pos)
        self.current_player = -1 * self.current_player
        self.turn = self.turn + 1
        
    
    # Display the status of the board
    def print_board(self):
        for i in range(BOARD_SIZE):
            print("-------------------------------")
            top_line = "|"
            mid_line = "|"
            bot_line = "|"
            for j in range(BOARD_SIZE):
                card = self.get_card(i, j)
                disp_num = lambda x: str(x) if x < 10 else 'A'
                top_line = top_line + ( "   |" if card is None else ( " " + disp_num(card.get_top()) + " |") )
                mid_line = mid_line + ( "   |" if card is None else (disp_num(card.get_left()) + " " + disp_num(card.get_right()) + "|") )
                bot_line = bot_line + ( "   |" if card is None else ( " " + disp_num(card.get_bottom()) + " |") )
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
    On Board: If the card has been placed on the board
    Name: the name of this card. Supposed to be unique.
    Rank: the rank of this card. If -1 it means this card is not classified
    Element: Only for special rules. Not implemented yet.
    '''
    
    def __init__(self, numbers, owner = NO_ONE, visible = False, position = (-1, -1), name = "", rank = -1,  element = -1):
        self.numbers = numbers
        self.owner = owner
        self.visible = visible
        self.position = position
        self.name = name
        self.rank = rank
        self.element = element
    
    def get_top(self):
        return self.get_number(TOP_INDEX)
    
    def get_right(self):
        return self.get_number(RIGHT_INDEX)
    
    def get_bottom(self):
        return self.get_number(BOTTOM_INDEX)
    
    def get_left(self):
        return self.get_number(LEFT_INDEX)
    
    def get_number(self, index):
        return self.numbers[index]
    
    def get_rank(self):
        return min(self.rank, MAX_RANK_LEVEL)
    
    def get_value(self):
        # To measure the value of a card. We use a basic logic here - the sum of four numbers of the card
        return sum(self.numbers)
    
    
    def reset(self, owner = NO_ONE, visible = False, position = (-1, -1)):
        # To improve the performance by resetting the owner/visible/position status, instead of creating new cards copies
        self.visible = visible
        self.owner = owner
        self.position = position


def load_cards_from_file(path, file_name):
    card_list = []
    with open(os.path.join(path,file_name), 'rt') as file:
        cards = csv.DictReader(file)
        for card in cards:
            card_list.append( Card(
                numbers = [ int(card['top']), int(card['right']), int(card['bottom']), int(card['left']) ],
                name = card['name'],
                rank = card['rank'],
                element = card['element']
                ) )   
    return card_list  
     