'''
Reward helper functions
'''

import numpy as np

M_IN_K = 1000.0
VIDEO_BIT_RATE = [300, 1200, 2850, 6500, 33000, 165000]  # Kbps


def linear_reward(args, bit_rate, rebuf, last_bit_rate):
    '''Generates a MORL vectorized reward based on bit rate, rebuffering, and last bit rate.

    The scalarized reward then becomes the dot product of a user's
    preference weights and the MORL reward vector.
    '''

    current_bitrate = VIDEO_BIT_RATE[bit_rate]
    last_bitrate = VIDEO_BIT_RATE[last_bit_rate]
    reward = current_bitrate / M_IN_K - args.REBUF_PENALTY * rebuf - \
             np.abs( current_bitrate - last_bitrate ) / M_IN_K

    return reward

