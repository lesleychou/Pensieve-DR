import itertools
import multiprocessing as mp

import numpy as np
from numba import jit

from utils.constants import (B_IN_MB, S_INFO, S_LEN, DEFAULT_QUALITY, M_IN_K,
                                MILLISECONDS_IN_SECOND, VIDEO_BIT_RATE,
                                VIDEO_CHUNK_LEN, REBUF_PENALTY, SMOOTH_PENALTY,
                                BITS_IN_BYTE, VIDEO_CHUNK_LEN, BITRATE_LEVELS,TOTAL_VIDEO_CHUNK,
                                PACKET_SIZE,NOISE_LOW, NOISE_HIGH, BUFFER_NORM_FACTOR,CHUNK_TIL_VIDEO_END_CAP)


class RobustMPC():
    """Naive implementation of RobustMPC."""

    def __init__(self, mpc_future_chunk_cnt=5, accumulate_past_error=False):
        self.mpc_future_chunk_cnt = mpc_future_chunk_cnt

        # all possible combinations of 5 chunk bitrates (9^5 options)
        # iterate over list and for each, compute reward and store max
        # reward combination
        self.chunk_combo_options = np.array(
            [combo for combo in itertools.product(
                range(len(VIDEO_BIT_RATE)), repeat=self.mpc_future_chunk_cnt)])
        self.bitrate_options = np.array(VIDEO_BIT_RATE)

        self.past_errors = []
        self.past_bandwidth_ests = []
        self.accumulate_past_error = accumulate_past_error

    def select_action(self, state, last_index, future_chunk_cnt, video_size,
                      bit_rate, buffer_size):
        # defualt assumes that this is the first request so error is 0
        # since we have never predicted bandwidth
        if (len(self.past_bandwidth_ests) > 0):
            self.past_errors.append(np.abs(
                self.past_bandwidth_ests[-1]-state[0, 2, -1])/state[0, 2, -1])
        else:
            self.past_errors.append(0)

        # pick bitrate according to MPC
        # first get harmonic mean of last 5 bandwidths
        past_bandwidths = state[0, 2, -5:]
        while past_bandwidths[0] == 0.0:
            past_bandwidths = past_bandwidths[1:]

        harmonic_bandwidth = 1 / np.mean(1 / past_bandwidths)
        print(harmonic_bandwidth, "------past_bandwidths")

        # future bandwidth prediction
        # divide by 1 + max of last 5 (or up to 5) errors
        max_error = 0
        error_pos = -5
        if (len(self.past_errors) < 5):
            error_pos = -len(self.past_errors)
        max_error = max(self.past_errors[error_pos:])
        future_bandwidth = harmonic_bandwidth / (1 + max_error)  # robustMPC
        self.past_bandwidth_ests.append(harmonic_bandwidth)

        bit_rate = predict_bitrate(
            future_chunk_cnt, buffer_size, bit_rate, last_index,
            future_bandwidth, video_size, self.chunk_combo_options,
            self.bitrate_options)
        return bit_rate, future_bandwidth

    def evaluate(self, net_env, all_file_names):
        """Evaluate on a single net_env."""
        results = []
        if not self.accumulate_past_error:
            self.past_bandwidth_ests = []
            self.past_errors = []

        net_env.reset()
        video_size = np.array([net_env.video_size[i]
                               for i in sorted(net_env.video_size)])
        time_stamp = 0
        bit_rate = DEFAULT_QUALITY
        future_bandwidth = 0

        while True:  # serve video forever
            # the action is from the last decision
            # this is to make the framework similar to the real
            state, reward, end_of_video, info = net_env.get_video_chunk(bit_rate)
            print(reward)

            time_stamp += info['delay']  # in ms
            time_stamp += info['sleep_time']  # in ms

            results.append([all_file_names[net_env.trace_idx],
                            time_stamp / M_IN_K,
                            self.bitrate_options[bit_rate],
                            info['buffer_size'], info['rebuf'],
                            info['video_chunk_size'], info['delay'], reward,
                            future_bandwidth])

            # future chunks length (try 4 if that many remaining)
            last_index = (net_env.total_video_chunk -
                          info['video_chunk_remain'] - 1)
            future_chunk_cnt = min(self.mpc_future_chunk_cnt,
                                   net_env.total_video_chunk - last_index - 1)

            bit_rate, future_bandwidth = self.select_action(
                state, last_index, future_chunk_cnt, video_size, bit_rate,
                info['buffer_size'])

            if end_of_video:
                if end_of_video:
                    next_trace = net_env.play_next()
                    if next_trace is None:
                        break
                    else:
                        pass
                        # print( "Evaluating against" ,next_trace.filename )
        return results

    def evaluate_envs(self, net_envs, n_proc=mp.cpu_count()):
        """Evaluate multipe environment using multiprocessing."""
        arguments = net_envs
        with mp.Pool(processes=n_proc) as pool:
            results = pool.starmap(self.evaluate, arguments)
        return results


@jit(nopython=True)
def predict_bitrate(future_chunk_length, buffer_size, bit_rate, last_index,
                    future_bandwidth, video_size, chunk_combo_options,
                    bitrate_options):
    print(future_bandwidth, "----bitrate_options")
    max_reward = np.NINF
    best_combo = ()
    start_buffer = buffer_size

    for full_combo in chunk_combo_options:
        combo = full_combo[0:int(future_chunk_length)]
        # calculate total rebuffer time for this combination (start with
        # start_buffer and subtract each download time and add 2 seconds in
        # that order)
        curr_buffer = start_buffer
        reward = 0
        last_quality = int(bit_rate)
        for position in range(0, len(combo)):
            chunk_quality = combo[position]
            # e.g., if last chunk is 3, then first iter is 3+0+1=4
            index = last_index + position + 1
            # this is MB/MB/s --> seconds
            row_len = len(video_size[chunk_quality])
            download_time = video_size[chunk_quality,
                                       int(index % (row_len-1))] / \
                B_IN_MB / future_bandwidth
            if (curr_buffer < download_time):
                rebuffer_time = (download_time - curr_buffer)
                curr_buffer = 0
            else:
                curr_buffer -= download_time
                rebuffer_time = 0
            curr_buffer += VIDEO_CHUNK_LEN / MILLISECONDS_IN_SECOND

            reward = bitrate_options[bit_rate] / M_IN_K - REBUF_PENALTY * rebuffer_time - \
                     SMOOTH_PENALTY * np.abs( bitrate_options[bit_rate] - bitrate_options[last_quality] ) / M_IN_K

            last_quality = chunk_quality
        # compute reward for this combination (one reward per 5-chunk combo)
        # bitrates are in Mbits/s, rebuffer in seconds, and smoothness_diffs in
        # Mbits/s

        if reward >= max_reward:
            best_combo = combo

            max_reward = reward
            # send data to html side (first chunk of best combo)
            # no combo had reward better than -1000000 (ERROR) so send 0
            send_data = 0
            if best_combo.size != 0:  # some combo was good
                send_data = best_combo[0]

    return send_data
