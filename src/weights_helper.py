"""Helper functions for dealing with multi objective reinforcement
learning weight preferences.
"""

import numpy as np

def generate_exp_weights(set_size):
    '''
    Returns a numpy array of a single sample, either 1.0 or 2.5, with a 50/50 chance.
    '''
    exp_w = np.array([np.random.choice([1.0, 2.5], p=[1.0, 0.0]) for _ in range(set_size)])
    return exp_w

def generate_weights_set_with_exp(args, set_size, reward_size, bitrate_exp=2.5, fixed=False):
    '''
    Generates random weights from parameters, including column for bitrate_exp (included as an argument).

    Sample output:

    array([[0.22975476, 0.17374657, 0.59649867, 2.5       ],
       [0.45385297, 0.42113295, 0.12501408, 2.5       ],
       [0.59160269, 0.05562816, 0.35276915, 2.5       ],
       [0.45215923, 0.22092943, 0.32691133, 2.5       ],
       [0.64912316, 0.15886377, 0.19201306, 2.5       ],
       [0.46046734, 0.0568026 , 0.48273006, 2.5       ],
       [0.47403606, 0.20254634, 0.3234176 , 2.5.        ],
       [0.08734331, 0.46332982, 0.44932687, 2.5.        ]])
    '''
    weights = generate_weights_set(args, set_size, reward_size, fixed=fixed)
    exp_w = np.array([bitrate_exp for _ in range(set_size)])
    exp_w = np.expand_dims(exp_w, 1)
    final = np.concatenate((weights, exp_w), axis=1)
    #print(final)
    return final

def generate_weights_set(args, set_size, reward_size, fixed=False):
    '''
    Generates random weights from parameters.

    num_preferences: The number of preferences to be generated. 
    reward_size: reward size

    Ex:

    >>> generate_weights(2, 3)
    [[0.23439905 0.65466162 0.11093933]
     [0.0350973  0.48822209 0.47668061]]
    '''
    if fixed is True:
        return np.tile(np.array([args.bitrate_weight, args.rebuf_weight]), (set_size, 1))

    weights = np.random.randn(set_size, reward_size)
    # Normalization
    weights = np.abs(weights) / np.linalg.norm(weights, ord=1, axis=1).reshape(set_size, 1)
    return weights

def sample_weight_vector(reward_size):
    '''
    Generates a single vector of weight preferences, matching the reward size.

    Ex:

    >>> generate_weights(2, 3)
    [0.23439905 0.65466162 0.11093933]
    '''
    return generate_weights_set(1, reward_size)[0]

'''
>>> generate_weights(2, 3)
    [[0.23439905 0.65466162 0.11093933, 1.0]
     [0.0350973  0.48822209 0.47668061, 2.5]]
'''
