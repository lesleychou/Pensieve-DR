import argparse
import os
from utils.utils import adjust_traces, load_traces
import a3c
# import fixed_env as env
import env
import numpy as np
import tensorflow as tf


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
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0
REBUF_PENALTY = 165  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
#RANDOM_SEED = 42
RAND_RANGE = 1000
# LOG_FILE = './test_results/log_sim_rl'
# TEST_TRACES = './cooked_test_traces/'
# TEST_TRACES = './test_sim_traces/'
# TEST_TRACES = '../data/val/'
# log in format of time_stamp bit_rate buffer_size rebuffer_time chunk_size download_time reward


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pensieve testing script.")
    parser.add_argument("--test_trace_dir", type=str,
                        required=True, help='dir to all test traces.')
    parser.add_argument("--summary_dir", type=str,
                        required=True, help='output path.')
    parser.add_argument("--model_path", type=str, required=True,
                        help='model path')
    parser.add_argument("--random_seed", type=int, default=11)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument( '--ROBUST_NOISE', type=float, default='0.1', help='' )

    parser.add_argument( '--SAMPLE_LENGTH', type=int, default='4', help='' )
    parser.add_argument( '--NUMBER_PICK', type=int, default='1', help='' )
    parser.add_argument( '--A_DIM', type=int, default='6', help='' )
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
    if bit_rate > 10:
        bit_rate = 10

    #print(bit_rate)
    return bit_rate

def main():
    args = parse_args()
    summary_dir = args.summary_dir
    nn_model = args.model_path
    test_trace_dir = args.test_trace_dir
    os.makedirs(summary_dir, exist_ok=True)
    #np.random.seed(args.random_seed)

    #assert len(VIDEO_BIT_RATE) == A_DIM

    all_cooked_time, all_cooked_bw, all_file_names = load_traces(
        test_trace_dir)
    #print(len(all_cooked_bw))
    #print(min(min(all_cooked_bw)), max(max(all_cooked_bw)))
    # all_cooked_time, all_cooked_bw = adjust_traces(
    #     all_cooked_time,
    #     all_cooked_bw,
    #     test_trace_dir,
    #     args.random_seed)

    # adjust_traces_one_random(all_ts, all_bw, random_seed, sample_length = 4):
    # all_cooked_time, all_cooked_bw = adjust_n_random_traces(
    #     all_cooked_time,
    #     all_cooked_bw,
    #     args.random_seed,
    #     args.ROBUST_NOISE,
    #     args.SAMPLE_LENGTH,
    #     args.NUMBER_PICK)


    #print(len(all_cooked_time[-1]))

    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw, fixed=True)

    log_path = os.path.join(summary_dir, 'log_sim_rl_' +
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
            bit_rate = (action_cumsum > np.random.randint(
                 1, RAND_RANGE) / float(RAND_RANGE)).argmax()
            #bit_rate = calculate_from_selection( selection ,last_bit_rate )
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
                action_vec[bit_rate] = 1

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


if __name__ == '__main__':
    main()
