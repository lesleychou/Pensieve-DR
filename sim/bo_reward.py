import argparse
import os
from utils.utils import adjust_traces, load_traces
import a3c
# import fixed_env as env
import env
import numpy as np
import tensorflow as tf
from statistics import mean
import itertools
from numba import jit

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

tf.logging.set_verbosity(tf.logging.INFO)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_INFO = 6
#S_LEN = 11  # take how many frames in the past
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
#VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300, 6500, 9800, 14700, 22050, 33000]  # Kbps
VIDEO_BIT_RATE = [300, 1200, 2850, 6500, 33000, 165000]  # Kbps

VIDEO_BIT_RATE_MPC = np.array([300, 1200, 2850, 6500, 33000, 165000])  # Kbps

BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0
REBUF_PENALTY = 165  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 0  # default video quality without agent
#RANDOM_SEED = 42
RAND_RANGE = 1000
# LOG_FILE = './test_results/log_sim_rl'
# TEST_TRACES = './cooked_test_traces/'
# TEST_TRACES = './test_sim_traces/'
# TEST_TRACES = '../data/val/'
# log in format of time_stamp bit_rate buffer_size rebuffer_time chunk_size download_time reward
_INFO = 5  # bit_rate, buffer_size, rebuffering_time, bandwidth_measurement, chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6
MPC_FUTURE_CHUNK_COUNT = 5
TOTAL_VIDEO_CHUNKS = 48
RANDOM_SEED = 42
CHUNK_COMBO_OPTIONS = []

# past errors in bandwidth
past_errors = []
past_bandwidth_ests = []
VIDEO_SIZE_FILE = '../data/video_size_6_larger/video_size_'


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pensieve testing script.")
    parser.add_argument("--test_trace_dir", type=str,
                        required=True, help='dir to all test traces.')
    parser.add_argument("--summary_dir", type=str,
                        required=True, help='output path.')
    parser.add_argument( "--mpc_summary_dir" ,type=str ,
                         required=True ,help='output path.' )
    parser.add_argument("--model_path", type=str, required=True,
                        help='model path')

    parser.add_argument( '--A_DIM', type=int, default='3', help='' )
    parser.add_argument( '--BITRATE_DIM', type=int, default='6', help='' )
    parser.add_argument( '--S_LEN', type=int, default='6', help='' )

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



def generate_traces_with(config):
    """
    Generates traces based on the config
    """
    script = "./environment/trace_generator_from_config.py"
    command = "python {script} \"{config}\"".format(script=script, config=vars(config))
    # alternatively call with os.system, but it doesn't print the result that way
    os.system(command)
    # output = subprocess.check_output(command, shell=True, text=True).strip()
    # print(output)


def main():
    args = parse_args()

    # Generate new trace by BO

    # test by RL and MPC
    summary_dir = args.summary_dir
    nn_model = args.model_path
    test_trace_dir = args.test_trace_dir
    os.makedirs(summary_dir, exist_ok=True)
    np.random.seed(RANDOM_SEED)


    all_cooked_time, all_cooked_bw, all_file_names = load_traces(
        test_trace_dir)

    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw, fixed=True)

    # log_path = os.path.join(summary_dir, 'log_sim_rl_' +
    #                         all_file_names[net_env.trace_idx])
    # log_file = open(log_path, 'w')

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
            print("Testing model restored.")

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
        results = []

        while True:  # serve video forever
            # the action is from the last decision
            # this is to make the framework similar to the real
            # TODO: remove this into the net_env.step
            # delay, sleep_time, buffer_size, rebuf, \
            #     video_chunk_size, next_video_chunk_sizes, \
            #     end_of_video, video_chunk_remain = \
            #     net_env.get_video_chunk(bit_rate)
            #
            state ,reward ,end_of_video ,info = net_env. get_video_chunk( bit_rate )

            time_stamp += info['delay']  # in ms
            time_stamp += info['sleep_time']  # in ms

            results.append( [all_file_names[net_env.trace_idx],
                             time_stamp / M_IN_K ,VIDEO_BIT_RATE[bit_rate] ,
                             info['buffer_size'] ,info['rebuf'] ,
                             info['video_chunk_size'] ,info['delay'] ,reward] )

            ### log the results:
            log_file.write( str( time_stamp / M_IN_K ) + '\t' +
                            str( VIDEO_BIT_RATE[bit_rate] ) + '\t' +
                            str(info['buffer_size']) + '\t' +
                            str(info['rebuf']) + '\t' +
                            str(info['video_chunk_size'])  + '\t' +
                            str(info['delay']) + '\t' +
                            str( reward ) + '\n' )
            log_file.flush()


            # retrieve previous state
            if len(s_batch) == 0:
                state = [np.zeros((S_INFO, args.S_LEN))]
            else:
                state = np.array(s_batch[-1], copy=True)

            s_batch.append(state)

            action_prob = actor.predict(np.reshape(state, (1, S_INFO, args.S_LEN)))
            action_cumsum = np.cumsum(action_prob)
            selection = (action_cumsum > np.random.randint(
                 1, RAND_RANGE) / float(RAND_RANGE)).argmax()
            bit_rate = calculate_from_selection( selection ,last_bit_rate )

            entropy_record.append(a3c.compute_entropy(action_prob[0]))

            if end_of_video:
                # log_file.write('\n')
                # log_file.close()
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

                log_path = os.path.join(
                    summary_dir,
                    'log_sim_rl_{}'.format(all_file_names[net_env.trace_idx]))
                log_file = open(log_path, 'w')

            test_dir = summary_dir
            plot_files = os.listdir( test_dir )

        # TODO: debug what's the right way to calculate the mean reward
        rl_avg_chunk_reward = means_from_trace_list(results)
        avg_chunk_reward = given_string_mean_reward(plot_files, test_dir, str='FCC')

        print(avg_chunk_reward)

        print(rl_avg_chunk_reward)

        # MPC = MPC_ref( test_result_dir=args.mpc_summary_dir ,test_trace_dir=args.test_trace_dir )
        # MPC.run()
        #
        # test_dir = args.mpc_summary_dir
        # plot_files_mpc = os.listdir( test_dir )
        # mpc_mean_reward = given_string_mean_reward( plot_files_mpc ,test_dir ,str='' )
        #
        # print( mpc_mean_reward-rl_mean_reward )

def given_string_mean_reward(plot_files ,test_dir ,str):
    matching = [s for s in plot_files if str in s]
    reward = []
    count = 0
    for log_file in matching:
        count += 1
        print(log_file)
        with open( test_dir +'/'+ log_file ,'r' ) as f:
            for line in f:
                parse = line.split()
                if len( parse ) <= 1:
                    break
                reward.append( float( parse[6] ) )
    print(count)
    return np.mean( reward )


def means_from_trace_list(chunks):
    """
    chunks: List of trace chunks with format

    [ ["trace1", ..., r], ["trace1", ..., r], ["trace2", ..., r], ["trace2", ..., r]]

    Groups the chunks by trace, removes the first chunk from each trace, and returns the means of the reward
    """
    # Group by trace name
    traces = {}
    for c in chunks:
        name = c[0]
        if name not in traces:
            traces[name] = []
        traces[name].append( c[-1] )
    means = np.mean( [np.mean( np.array( traces[key][1:] ) ) for idx ,key in enumerate( traces )] )
    return means

if __name__ == '__main__':
    main()
