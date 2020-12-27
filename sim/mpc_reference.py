import argparse
import itertools
import os
from numba import jit
#from pensieve.trace_generator import TraceGenerator

# TODO: Merge utils.env to Pensieve.env?
import env as env
import numpy as np
from utils.utils import load_traces

S_INFO = 5  # bit_rate, buffer_size, rebuffering_time, bandwidth_measurement, chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6
MPC_FUTURE_CHUNK_COUNT = 5

VIDEO_BIT_RATE = np.array([300, 1200, 2850, 6500, 33000, 165000])  # Kbps
BITRATE_REWARD = [1, 2, 3, 12, 15, 20]
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
TOTAL_VIDEO_CHUNKS = 48
M_IN_K = 1000.0
REBUF_PENALTY = 165  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 20

CHUNK_COMBO_OPTIONS = []
past_errors = []
past_bandwidth_ests = []
VIDEO_SIZE_FILE = '../data/video_size_6_larger/video_size_'
TEST_RESULT = '../results/mpc-UDR-val'
TEST_TRACE = '../data/generated_traces_UDR/val_0_20000/'

@jit(nopython=True)
def get_chunk_size(quality, index, size_video_array):
    if (index < 0 or index > 48):
        return 0
    # note that the quality and video labels are inverted (i.e., quality 4 is
    # highest and this pertains to video1)
    return size_video_array[quality, index]

@jit(nopython=True)
def calculate_rebuffer(size_video_array, future_chunk_length, buffer_size, bit_rate, last_index, future_bandwidth, CHUNK_COMBO_OPTIONS):
    max_reward = -100000000
    start_buffer = buffer_size

    for full_combo in CHUNK_COMBO_OPTIONS:
        combo = full_combo[0:future_chunk_length]
        # calculate total rebuffer time for this combination (start with start_buffer and subtract
        # each download time and add 2 seconds in that order)
        curr_rebuffer_time = 0
        curr_buffer = start_buffer
        bitrate_sum = 0
        smoothness_diffs = 0
        last_quality = int( bit_rate )
        for position in range( 0, len( combo ) ):
            chunk_quality = combo[position]
            # e.g., if last chunk is 3, then first iter is 3+0+1=4
            index = last_index + position + 1
            # this is MB/MB/s --> seconds
            download_time = (get_chunk_size(chunk_quality, index, size_video_array) / 1000000.) / future_bandwidth
            if (curr_buffer < download_time):
                curr_rebuffer_time += (download_time - curr_buffer)
                curr_buffer = 0
            else:
                curr_buffer -= download_time
            curr_buffer += 4
            bitrate_sum += VIDEO_BIT_RATE[chunk_quality]
            smoothness_diffs += abs(
                VIDEO_BIT_RATE[chunk_quality] - VIDEO_BIT_RATE[last_quality] )
            last_quality = chunk_quality

        reward = (bitrate_sum / 1000.) - (REBUF_PENALTY *
                                          curr_rebuffer_time) - (smoothness_diffs / 1000.)
        if reward >= max_reward:
            best_combo = combo
            max_reward = reward
            send_data = 0
            if best_combo.size != 0:  # some combo was good
                send_data = best_combo[0]
    return send_data


class MPC_ref(object):
    def __init__(self, test_result_dir, test_trace_dir):
        '''
        Run MPC as the reference environment
        :param test_result_dir: log the MPC test result
        :param test_trace_dir: trace generated by Randomized-Trace Generator
        '''
        self.summary_dir = test_result_dir
        self.test_dir = test_trace_dir

    def run(self):
        summary_dir = self.summary_dir
        os.makedirs(summary_dir, exist_ok=True)
        np.random.seed(RANDOM_SEED)

        size_video_array =[]
        for bitrate in range( 6 ):
            video_size = []
            with open( VIDEO_SIZE_FILE + str( bitrate ) ) as f:
                for line in f:
                    video_size.append( int( line.split()[0] ) )
            size_video_array.append(video_size)

        size_video_array = np.array(np.squeeze(size_video_array))

        assert len(VIDEO_BIT_RATE) == A_DIM
        print(self.test_dir)

        # TODO: let TraceGenerator.generate_trace return all_file_names, and write traces out?
        all_cooked_time ,all_cooked_bw ,all_file_names = load_traces(self.test_dir)
        #all_cooked_time, all_cooked_bw, all_file_names = TraceGenerator.generate_trace()

        net_env = env.Environment(all_cooked_time=all_cooked_time,
                                  all_cooked_bw=all_cooked_bw, fixed=True)

        log_path = os.path.join(
            summary_dir, 'log_sim_mpc_' + all_file_names[net_env.trace_idx])
        log_file = open(log_path, 'w', 1)

        time_stamp = 0

        last_bit_rate = DEFAULT_QUALITY
        bit_rate = DEFAULT_QUALITY

        action_vec = np.zeros(A_DIM)
        action_vec[bit_rate] = 1

        s_batch = [np.zeros((S_INFO, S_LEN))]
        a_batch = [action_vec]
        r_batch = []

        video_count = 0

        # make chunk combination options
        for combo in itertools.product([0, 1, 2, 3, 4, 5], repeat=5):
            CHUNK_COMBO_OPTIONS.append(combo)

        while True:  # serve video forever
            # the action is from the last decision
            # this is to make the framework similar to the real
            delay, sleep_time, buffer_size, rebuf, \
                video_chunk_size, next_video_chunk_sizes, \
                end_of_video, video_chunk_remain = \
                net_env.get_video_chunk(bit_rate)

            time_stamp += delay  # in ms
            time_stamp += sleep_time  # in ms

            # reward is video quality - rebuffer penalty
            reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
                - REBUF_PENALTY * rebuf \
                - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                          VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K


            r_batch.append(reward)

            last_bit_rate = bit_rate

            # log time_stamp, bit_rate, buffer_size, reward
            log_file.write(str(time_stamp / M_IN_K) + '\t' +
                           str(VIDEO_BIT_RATE[bit_rate]) + '\t' +
                           str(buffer_size) + '\t' +
                           str(rebuf) + '\t' +
                           str(video_chunk_size) + '\t' +
                           str(delay) + '\t' +
                           str(reward) + '\n')

            # retrieve previous state
            if len(s_batch) == 0:
                state = [np.zeros((S_INFO, S_LEN))]
            else:
                state = np.array(s_batch[-1], copy=True)

            # dequeue history record
            state = np.roll(state, -1, axis=1)

            # this should be S_INFO number of terms
            state[0, -1] = VIDEO_BIT_RATE[bit_rate] / \
                float(np.max(VIDEO_BIT_RATE))  # last quality
            state[1, -1] = buffer_size / BUFFER_NORM_FACTOR
            state[2, -1] = rebuf
            state[3, -1] = float(video_chunk_size) / \
                float(delay) / M_IN_K  # kilo byte / ms
            state[4, -1] = np.minimum(video_chunk_remain,
                                      CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)
            # state[5: 10, :] = future_chunk_sizes / M_IN_K / M_IN_K

            # ================== MPC =========================
            curr_error = 0  # defualt assumes that this is the first request so error is 0 since we have never predicted bandwidth
            if (len(past_bandwidth_ests) > 0):
                curr_error = abs(
                    past_bandwidth_ests[-1]-state[3, -1])/float(state[3, -1])
            past_errors.append(curr_error)

            # pick bitrate according to MPC
            # first get harmonic mean of last 5 bandwidths
            past_bandwidths = state[3, -5:]
            while past_bandwidths[0] == 0.0:
                past_bandwidths = past_bandwidths[1:]

            bandwidth_sum = 0
            for past_val in past_bandwidths:
                bandwidth_sum += (1/float(past_val))
            harmonic_bandwidth = 1.0/(bandwidth_sum/len(past_bandwidths))

            error_pos = -5
            if (len(past_errors) < 5):
                error_pos = -len(past_errors)
            max_error = float(max(past_errors[error_pos:]))
            future_bandwidth = harmonic_bandwidth/(1+max_error)  # robustMPC here
            past_bandwidth_ests.append(harmonic_bandwidth)

            # future chunks length (try 4 if that many remaining)
            last_index = int(CHUNK_TIL_VIDEO_END_CAP - video_chunk_remain)
            future_chunk_length = MPC_FUTURE_CHUNK_COUNT
            if (TOTAL_VIDEO_CHUNKS - last_index < 5):
                future_chunk_length = TOTAL_VIDEO_CHUNKS - last_index

            # all possible combinations of 5 chunk bitrates (9^5 options)
            # iterate over list and for each, compute reward and store max reward combination
            # start = time.time()
            chunk_combo_options = np.array( CHUNK_COMBO_OPTIONS )

            bit_rate = calculate_rebuffer(size_video_array, future_chunk_length, buffer_size, bit_rate,
                                          last_index, future_bandwidth, chunk_combo_options)

            s_batch.append(state)

            if end_of_video:
                log_file.write('\n')
                log_file.close()

                last_bit_rate = DEFAULT_QUALITY
                bit_rate = DEFAULT_QUALITY  # use the default action here

                del s_batch[:]
                del a_batch[:]
                del r_batch[:]

                action_vec = np.zeros(A_DIM)
                action_vec[bit_rate] = 1

                s_batch.append(np.zeros((S_INFO, S_LEN)))
                a_batch.append(action_vec)

                print( "video count", video_count )
                video_count += 1

                if video_count >= len( all_file_names ):
                    break

                log_path = os.path.join(
                    summary_dir,
                    'log_sim_mpc_' + all_file_names[net_env.trace_idx])
                log_file = open(log_path, 'w', 1)
                #print("mpc test done on", all_file_names[net_env.trace_idx])

def given_string_mean_reward(plot_files ,test_dir ,str):
    matching = [s for s in plot_files if str in s]
    reward = []
    count=0
    for log_file in matching:
        count+=1
        print(log_file)
        with open( test_dir +'/'+ log_file ,'r' ) as f:
            for line in f:
                parse = line.split()
                if len( parse ) <= 1:
                    break
                reward.append( float( parse[6] ) )
    #print(count)
    mean = np.mean( reward[1:] )
    return round(mean, 2)


def main():
    # MPC = MPC_ref(test_result_dir=TEST_RESULT, test_trace_dir=TEST_TRACE)
    # MPC.run()

    test_dir = TEST_RESULT
    plot_files = os.listdir( test_dir )

    # reward_0 = given_string_mean_reward( plot_files ,test_dir ,str='BW_0-150' )
    # reward_1 = given_string_mean_reward( plot_files ,test_dir ,str='BW_150-250' )
    # reward_2 = given_string_mean_reward( plot_files ,test_dir ,str='BW_250-350' )
    # reward_3 = given_string_mean_reward( plot_files ,test_dir ,str='BW_350-450' )
    # reward_4 = given_string_mean_reward( plot_files ,test_dir ,str='BW_450-550' )
    #
    # mpc_mean_reward = {'0-150': reward_0 ,
    #                   '150-250': reward_1 ,
    #                   '250-350': reward_2 ,
    #                   '350-450': reward_3 ,
    #                   '450-550': reward_4}

    mpc_mean_reward = {}
    bw_range ,count = 1000000 ,50
    for i in range( count ):
        step = bw_range / count
        low = round( step * i )
        high = round( step * (i + 1) )
        dir = str( low ) + '_' + str( high )
        print( dir )
        mpc_mean_reward[dir] = given_string_mean_reward( plot_files ,test_dir ,str=dir )

    print( mpc_mean_reward ,"-----mpc_mean_reward-----" )


if __name__ == '__main__':
    main()