'''
State helpers
'''

import numpy as np
import torch
import src.config as args


VIDEO_BIT_RATE = [300, 1200, 2850, 6500, 33000, 165000]  # Kbps
M_IN_K = 1000.0

def update_state_history(args,
                         action_dim,
                         state_history,
                         bit_rate,
                         delay,
                         buffer_size,
                         video_chunk_size,
                         next_video_chunk_sizes,
                         video_chunk_remain):
    '''The state is a running history of what is occuring. To update the
    state history, shift all relevant values and add the newest values
    in the most recent time slot.
    '''

    # Dequeue the oldest part of the state out of the history.
    #
    # Example:
    # tensor([[[0., 0., 0., 0., 0., 0., 0., 1.],
    #          [0., 0., 0., 0., 0., 0., 0., 2.],
    #          [0., 0., 0., 0., 0., 0., 0., 3.],
    #          [0., 0., 0., 0., 0., 0., 0., 0.],
    #          [0., 0., 0., 0., 0., 0., 0., 0.],
    #          [0., 0., 0., 0., 0., 0., 0., 0.]]])
    #
    # -->
    #
    # tensor([[[0., 0., 0., 0., 0., 0., 1., 0.],
    #          [0., 0., 0., 0., 0., 0., 2., 0.],
    #          [0., 0., 0., 0., 0., 0., 3., 0.],
    #          [0., 0., 0., 0., 0., 0., 0., 0.],
    #          [0., 0., 0., 0., 0., 0., 0., 0.],
    #          [0., 0., 0., 0., 0., 0., 0., 0.]]])
    state_history = torch.roll(state_history, -1, dims=-1)

    # last quality
    state_history[0, 0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))
    state_history[0, 1, -1] = buffer_size / args.BUFFER_NORM_FACTOR  # 10 sec
    state_history[0, 2, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
    state_history[0, 3, -1] = float(delay) / M_IN_K / args.BUFFER_NORM_FACTOR  # 10 sec
    state_history[0, 4, :action_dim] = torch.tensor(next_video_chunk_sizes).to(device=args.device) / M_IN_K / M_IN_K  # mega byte
    state_history[0, 5, -1] = min(video_chunk_remain, args.CHUNK_TIL_VIDEO_END_CAP) / float(args.CHUNK_TIL_VIDEO_END_CAP)

    return state_history
