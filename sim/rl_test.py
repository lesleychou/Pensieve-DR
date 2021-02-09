import argparse
import os
from utils.utils import adjust_traces, load_traces
import a3c
# import fixed_env as env
import env
import numpy as np
import tensorflow as tf
import subprocess
import itertools
import os
from numba import jit

S_INFO_MPC = 5  # bit_rate, buffer_size, rebuffering_time, bandwidth_measurement, chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6
MPC_FUTURE_CHUNK_COUNT = 5
VIDEO_BIT_RATE_MPC = np.array([300, 1200, 2850, 6500, 33000, 165000])  # Kbps
TOTAL_VIDEO_CHUNKS = 48
CHUNK_COMBO_OPTIONS = []
past_errors = []
past_bandwidth_ests = []
VIDEO_SIZE_FILE = '../data/video_size_6_larger/video_size_'

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

tf.logging.set_verbosity(tf.logging.INFO)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_INFO = 6
#S_LEN = 11  # take how many frames in the past
VIDEO_BIT_RATE = [300, 1200, 2850, 6500, 33000, 165000]  # Kbps
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0
REBUF_PENALTY = 165  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 0  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000


# Strategy:

# Input for RL Testing should be:
#
# 1. a configuration from which test traces are generated
#   - load the configuration from json and create a TraceConfig to generate traces (later)
#   - create the traces from a configuration (refer to example) (priority)
#
# 2. a model checkpoint file to load and test against the traces (DONE)
# 3. Move TraceConfig outside of this file so it can be used elsewhere too. (later)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Pensieve testing script.")
    parser.add_argument("--test_trace_dir", type=str,
                        # optional now because we have a default example
                        # required=True,
                        help='dir to generate all test traces.')
    parser.add_argument("--summary_dir", type=str,
                        required=True, help='output path.')
    parser.add_argument("--model_path", type=str, required=True,
                        help='model path')

    parser.add_argument( '--A_DIM', type=int, default='3', help='' )
    parser.add_argument( '--BITRATE_DIM', type=int, default='6', help='' )
    parser.add_argument( '--S_LEN', type=int, default='6', help='' )

    parser.add_argument( '--CURRENT_PARAM', type=float, default='10', help='the max-BW param BO input' )



    return parser.parse_args()

def calculate_from_selection(selected, last_bit_rate):
    # selected_action is 0-5
    # naive step implementation
    if selected == 1:
        bit_rate = last_bit_rate
    elif selected == 2:
        bit_rate = last_bit_rate + 1
    else:
        bit_rate = last_bit_rate - 1
    # bound
    if bit_rate < 0:
        bit_rate = 0
    if bit_rate > 5:
        bit_rate = 5

    #print(bit_rate)
    return bit_rate

def given_string_mean_reward(plot_files ,test_dir ,str):
    matching = [s for s in plot_files if str in s]
    reward = []
    count=0
    for log_file in matching:
        count+=1
        #print(log_file)
        with open( test_dir +'/'+ log_file ,'r' ) as f:
            for line in f:
                parse = line.split()
                if len( parse ) <= 1:
                    break
                reward.append( float( parse[6] ) )
    return np.mean( reward[1:] )

class TraceConfig:
    def __init__(self,
                 trace_dir,                 
                 max_throughput=10):
        self.trace_dir = trace_dir
        self.max_throughput = max_throughput
        self.T_l = 0
        self.T_s = 3
        self.cov = 3
        self.duration = 250
        self.step = 0
        self.min_throughput = 0.2
        self.num_traces = 100

def example_trace_config(args):
    return TraceConfig(args.test_trace_dir, max_throughput=args.CURRENT_PARAM)
        
def generate_traces_with(config):
    """
    Generates traces based on the config
    """
    script = "trace_generator.py"
    command = "python {script} \"{config}\"".format(script=script, config=vars(config))
    # alternatively call with os.system, but it doesn't print the result that way
    os.system(command)
    #output = subprocess.check_output(command, shell=True, text=True).strip()
    # print(output)


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
            bitrate_sum += VIDEO_BIT_RATE_MPC[chunk_quality]
            smoothness_diffs += abs(
                VIDEO_BIT_RATE_MPC[chunk_quality] - VIDEO_BIT_RATE_MPC[last_quality] )
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
        all_cooked_time ,all_cooked_bw ,all_file_names = load_traces(self.test_dir)

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

        s_batch = [np.zeros((S_INFO_MPC, S_LEN))]
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
            reward = VIDEO_BIT_RATE_MPC[bit_rate] / M_IN_K \
                - REBUF_PENALTY * rebuf \
                - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE_MPC[bit_rate] -
                                          VIDEO_BIT_RATE_MPC[last_bit_rate]) / M_IN_K
            r_batch.append(reward)

            last_bit_rate = bit_rate

            # log time_stamp, bit_rate, buffer_size, reward
            log_file.write(str(time_stamp / M_IN_K) + '\t' +
                           str(VIDEO_BIT_RATE_MPC[bit_rate]) + '\t' +
                           str(buffer_size) + '\t' +
                           str(rebuf) + '\t' +
                           str(video_chunk_size) + '\t' +
                           str(delay) + '\t' +
                           str(reward) + '\n')

            # retrieve previous state
            if len(s_batch) == 0:
                state = [np.zeros((S_INFO_MPC, S_LEN))]
            else:
                state = np.array(s_batch[-1], copy=True)

            # dequeue history record
            state = np.roll(state, -1, axis=1)

            # this should be S_INFO number of terms
            state[0, -1] = VIDEO_BIT_RATE_MPC[bit_rate] / \
                float(np.max(VIDEO_BIT_RATE_MPC))  # last quality
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
                # log_file.write('\n')
                # log_file.close()

                last_bit_rate = DEFAULT_QUALITY
                bit_rate = DEFAULT_QUALITY  # use the default action here

                del s_batch[:]
                del a_batch[:]
                del r_batch[:]

                action_vec = np.zeros(A_DIM)
                action_vec[bit_rate] = 1

                s_batch.append(np.zeros((S_INFO_MPC, S_LEN)))
                a_batch.append(action_vec)

                #print( "video count", video_count )
                video_count += 1

                if video_count >= len( all_file_names ):
                    break

                log_path = os.path.join(
                    summary_dir,
                    'log_sim_mpc_' + all_file_names[net_env.trace_idx])
                log_file = open(log_path, 'w', 1)

def main():
    args = parse_args()
    summary_dir = args.summary_dir
    nn_model = args.model_path

    # generate test traces 
    # test_trace_dir = args.test_trace_dir

    # Just manually load the example .... as an example...
    trace_config = example_trace_config(args)
    generate_traces_with(trace_config)
    
    np.random.seed(RANDOM_SEED)

    all_cooked_time, all_cooked_bw, all_file_names = load_traces(trace_config.trace_dir)

    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw, fixed=True)

    rl_summary_dir = summary_dir + '/' + 'rl_test'
    os.makedirs(rl_summary_dir, exist_ok=True)


    log_path = os.path.join(rl_summary_dir, 'log_sim_rl_' +
                            all_file_names[net_env.trace_idx])
    log_file = open(log_path, 'w')

    with tf.Session() as sess:

        actor = a3c.ActorNetwork(sess,
                                 state_dim=[S_INFO, args.S_LEN], action_dim=args.A_DIM,
                                 bitrate_dim=args.BITRATE_DIM
                                 )

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()  # save neural net parameters

        # restore neural net parameters
        if nn_model is not None:  # NN_MODEL is the path to file
            saver.restore(sess, nn_model)
            #print("Testing model restored.")

        time_stamp = 0

        last_bit_rate = DEFAULT_QUALITY
        bit_rate = DEFAULT_QUALITY

        action_vec = np.zeros(args.A_DIM)
        action_vec[bit_rate] = 1

        s_batch = [np.zeros((S_INFO, args.S_LEN))]
        a_batch = [action_vec]
        r_batch = []
        entropy_record = []

        video_count = 0

        while True:  # serve video forever
            # the action is from the last decision
            # this is to make the framework similar to the real
            delay, sleep_time, buffer_size, rebuf, \
                video_chunk_size, next_video_chunk_sizes, \
                end_of_video, video_chunk_remain = \
                net_env.get_video_chunk(bit_rate)

            time_stamp += delay  # in ms
            time_stamp += sleep_time  # in ms

            # reward is video quality - rebuffer penalty - smoothness
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
            log_file.flush()

            # retrieve previous state
            if len(s_batch) == 0:
                state = [np.zeros((S_INFO, args.S_LEN))]
            else:
                state = np.array(s_batch[-1], copy=True)

            # dequeue history record
            state = np.roll(state, -1, axis=1)

            # this should be S_INFO number of terms
            state[0, -1] = VIDEO_BIT_RATE[bit_rate] / \
                float(np.max(VIDEO_BIT_RATE))  # last quality
            state[1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
            state[2, -1] = float(video_chunk_size) / \
                float(delay) / M_IN_K  # kilo byte / ms
            state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
            state[4, :args.BITRATE_DIM] = np.array(
                next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
            state[5, -1] = np.minimum(
                video_chunk_remain,
                CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)

            action_prob = actor.predict(np.reshape(state, (1, S_INFO, args.S_LEN)))
            action_cumsum = np.cumsum(action_prob)
            selection = (action_cumsum > np.random.randint(
                 1, RAND_RANGE) / float(RAND_RANGE)).argmax()
            bit_rate = calculate_from_selection( selection ,last_bit_rate )
            # Note: we need to discretize the probability into 1/RAND_RANGE steps,
            # because there is an intrinsic discrepancy in passing single state and batch states

            s_batch.append(state)

            entropy_record.append(a3c.compute_entropy(action_prob[0]))

            if end_of_video:
                log_file.write('\n')
                log_file.close()

                last_bit_rate = DEFAULT_QUALITY
                bit_rate = DEFAULT_QUALITY  # use the default action here

                del s_batch[:]
                del a_batch[:]
                del r_batch[:]

                action_vec = np.zeros(args.A_DIM)
                action_vec[selection] = 1

                s_batch.append(np.zeros((S_INFO, args.S_LEN)))
                a_batch.append(action_vec)
                entropy_record = []

                video_count += 1

                if video_count >= len(all_file_names):
                    break

                log_path = os.path.join(rl_summary_dir,
                    'log_sim_rl_{}'.format(all_file_names[net_env.trace_idx]))
                log_file = open(log_path, 'w')

            test_dir = rl_summary_dir
            plot_files = os.listdir( test_dir )

        reward_0 = given_string_mean_reward( plot_files ,test_dir ,str='' )
        rl_mean_reward = reward_0

        mpc_summary_dir = summary_dir + '/' + 'mpc_test'
        os.makedirs( mpc_summary_dir ,exist_ok=True )

        MPC = MPC_ref( test_result_dir= mpc_summary_dir ,test_trace_dir=trace_config.trace_dir )
        MPC.run()

        test_dir_mpc = mpc_summary_dir
        plot_files_mpc = os.listdir( test_dir_mpc )
        reward_0_mpc = given_string_mean_reward( plot_files_mpc ,test_dir_mpc ,str='' )
        mpc_mean_reward = reward_0_mpc
        bo_reward = mpc_mean_reward - rl_mean_reward
        print(bo_reward)



if __name__ == '__main__':
    main()
