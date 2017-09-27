import numpy as np


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
    