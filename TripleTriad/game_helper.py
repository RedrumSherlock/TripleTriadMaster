import numpy as np
import time


def tuple2idx(board_size, x_pos, y_pos):
    return x_pos + y_pos * board_size

def idx2tuple(idx, board_size):
    return (idx % board_size, int(np.floor(idx/board_size)))

def vector2random_one_hot(vector):
    # The vector would be like [0, 0, 1, 0, 1, 0, 1, 0, 0]]
    # This function randomly leave one as 1 and the others as zero
    if vector.count(1) == 0:
        raise ValueError("The vector must have at least one valid option!")
    array = np.array(vector)
    choice = np.random.choice(np.where(array == 1)[0])
    
    one_hot = [0] * len(vector)
    one_hot[choice] = 1
    return one_hot
    
def indices2onehot(card_index, board_index, BOARD_SIZE, START_HANDS):
    cards = np.zeros(2 * START_HANDS)
    board = np.zeros(BOARD_SIZE ** 2)
    cards[card_index] = 1
    board[board_index] = 1
    return np.concatenate([cards, board], axis=0).reshape(1, 2 * START_HANDS + BOARD_SIZE ** 2)

def is_corner(x_pos, y_pos):
    return x_pos != 1 and y_pos != 1

def is_side(x_pos, y_pos):
    return (x_pos == 1 or y_pos == 1) and not (x_pos == 1 and y_pos == 1)

def timer(func):
    def timing(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print("{} ran for {} seconds".format(func.__name__, end - start))
        return result
    
    return timing