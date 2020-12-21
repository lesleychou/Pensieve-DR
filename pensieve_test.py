'''
Tests a specified actor model that outputs a policy.

Usage: python test.py <actor_model>
Example: python test.py model/actor.pt
'''

import random
import os

import torch
import numpy as np
from torch.distributions import Categorical
import torch.nn.functional as functional
import load_trace
import fixed_env as env

from src import reward as mor
from src import video_state
from src.model import model_Pensieve as model
import src.config as config

VIDEO_BIT_RATE = [300, 1200, 2850, 6500, 33000, 165000]  # Kbps
M_IN_K = 1000.0

TEST_TRACES = './data/generated_traces/val'

# log in format of time_stamp bit_rate buffer_size rebuffer_time chunk_size download_time reward

def main(args):
    #print(args)
    test_dir = args.TEST_RESULTS_DIR
    torch.set_num_threads(1)

    os.environ['PYTHONHASHSEED'] = str(args.RANDOM_SEED)
    random.seed(args.RANDOM_SEED)    
    np.random.seed(args.RANDOM_SEED)
    torch.manual_seed(args.RANDOM_SEED)

    assert len(VIDEO_BIT_RATE) == args.A_DIM

    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(TEST_TRACES)
    net_env = env.Environment(all_cooked_time=all_cooked_time, all_cooked_bw=all_cooked_bw)

    log_path = test_dir + "log_sim_rl" + '_' + all_file_names[net_env.trace_idx]
    log_file = open(log_path, 'w')

    net = model.ActorNetwork([args.S_INFO, args.S_LEN], args.A_DIM)
    net.load_state_dict(torch.load(args.ACTOR_FILEPATH))
    net.to(args.device)
    net.eval()

    # Initial setup
    time_stamp = 0
    last_bit_rate = args.DEFAULT_QUALITY
    bit_rate = args.DEFAULT_QUALITY
    start_of_video = True
    video_count = 0
    state = torch.zeros((args.S_INFO, args.S_LEN)).to(device=args.device)

    while True:  # serve video forever
        # the action is from the last decision
        # this is to make the framework similar to the real
        delay, sleep_time, buffer_size, rebuf, \
        video_chunk_size, next_video_chunk_sizes, \
        end_of_video, video_chunk_remain = \
            net_env.get_video_chunk(bit_rate)

        time_stamp += delay  # in ms
        time_stamp += sleep_time  # in ms

        reward = mor.linear_reward( args ,bit_rate ,rebuf ,last_bit_rate )

        last_bit_rate = bit_rate

        log_file.write(str(time_stamp / M_IN_K) + '\t' +
                       str(VIDEO_BIT_RATE[bit_rate]) + '\t' +
                       str(buffer_size) + '\t' +
                       str(rebuf) + '\t' +
                       str(video_chunk_size) + '\t' +
                       str(delay) + '\t' +
                       str(reward) + '\n')
        log_file.flush()

        # retrieve previous state
        state = torch.roll(state,-1,dims=-1)

        # this should be args.S_INFO number of terms
        state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))  # last quality
        state[1, -1] = buffer_size / args.BUFFER_NORM_FACTOR  # 10 sec
        state[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
        state[3, -1] = float(delay) / M_IN_K / args.BUFFER_NORM_FACTOR  # 10 sec
        state[4, :args.A_DIM] = torch.tensor(next_video_chunk_sizes).to(device=args.device) / M_IN_K / M_IN_K  # mega byte
        state[5, -1] = min(video_chunk_remain, args.CHUNK_TIL_VIDEO_END_CAP) / float(args.CHUNK_TIL_VIDEO_END_CAP)

        with torch.no_grad():
            probability = net.forward(state.unsqueeze(0))
            m = Categorical(probability)
            bit_rate = m.sample().item()

        if end_of_video:
            log_file.write('\n')
            log_file.close()

            last_bit_rate = args.DEFAULT_QUALITY
            bit_rate = args.DEFAULT_QUALITY  # use the default action here

            state = torch.zeros((args.S_INFO, args.S_LEN)).to(device=args.device)

            video_count += 1

            if video_count >= len(all_file_names):
                break

            log_path = test_dir + "log_sim_rl" + '_' + all_file_names[net_env.trace_idx]
            log_file = open(log_path, 'w')


if __name__ == '__main__':
    args = config.parse_args()
    args.device = torch.device('cpu')    
    main(args)
