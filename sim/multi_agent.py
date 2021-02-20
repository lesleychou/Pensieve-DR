import csv
import logging
import multiprocessing as mp
import os
import time

import a3c
import env
import numpy as np
import tensorflow as tf

import visdom
import src.config as config

from utils.utils import adjust_traces, load_traces
from datetime import datetime


# Visdom Settings
vis = visdom.Visdom()
assert vis.check_connection()
PLOT_COLOR = 'red'

tf.logging.set_verbosity(tf.logging.INFO)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ['CUDA_VISIBLE_DEVICES'] = ""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and
# time), chunk_til_video_end
MODEL_SAVE_INTERVAL = 100
#VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300, 6500, 9800, 14700, 22050, 33000]  # Kbps
#VIDEO_BIT_RATE = [300,750,1200,1850,2850,4300]  # Kbps
VIDEO_BIT_RATE = [300, 1200, 2850, 6500, 33000, 165000]

HD_REWARD = [1, 2, 3, 12, 15, 20]
M_IN_K = 1000.0
REBUF_PENALTY = 165  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 0  # default video quality without agent
NOISE = 0
DURATION = 1

# Env params need to do UDR
BUFFER_THRESH = 5000.0     # 60.0 * MILLISECONDS_IN_SECOND, max buffer limit
DRAIN_BUFFER_SLEEP_TIME = 500.0    # millisec
PACKET_PAYLOAD_PORTION = 0.6
LINK_RTT = 80  # millisec
LINK_RTT_MIN = 10
LINK_RTT_MAX = 330

UPDATE_ENV_INTERVAL = 1000


def calculate_from_selection(selected, last_bit_rate):
    # naive step implementation
    # action=0, bitrate-1; action=1, bitrate stay; action=2, bitrate+1
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

    return bit_rate


def entropy_weight_decay_func(epoch):
    # linear decay
    #return np.maximum(-0.05/(10**4) * epoch + 0.5, 0.1)
    return 0.5


def learning_rate_decay_func(epoch):
    if epoch < 20000:
        rate = 0.0001
    else:
        rate = 0.0001

    return rate


def test_on_env_params(args, params_dict, all_cooked_time, all_cooked_bw, all_file_names, actor, log_output_dir):
    # load env params from the paras_dict
    net_env = env.Environment( buffer_thresh=params_dict['buffer_thresh'],
                               drain_buffer_sleep_time=params_dict['drain_buffer_sleep_time'] ,
                               packet_payload_portion=params_dict['packet_payload_portion'],
                               link_rtt=LINK_RTT ,
                               all_cooked_time=all_cooked_time ,
                               all_cooked_bw=all_cooked_bw ,
                               all_file_names=all_file_names ,
                               fixed=True
                               )

    log_path = os.path.join( log_output_dir ,'log_sim_rl_{}'.format(
        all_file_names[net_env.trace_idx] ) )
    log_file = open( log_path ,'w' )

    time_stamp = 0

    last_bit_rate = DEFAULT_QUALITY
    bit_rate = DEFAULT_QUALITY
    selection = 0

    # action_vec = np.array( [VIDEO_BIT_RATE[last_bit_rate], VIDEO_BIT_RATE[bit_rate] ,selection] )
    action_vec = np.zeros( args.A_DIM )
    action_vec[selection] = 1

    s_batch = [np.zeros( (args.S_INFO ,args.S_LEN) )]
    a_batch = [action_vec]
    r_batch = []
    entropy_record = []

    video_count = 0

    while True:  # serve video forever
        # the action is from the last decision
        # this is to make the framework similar to the real
        delay ,sleep_time ,buffer_size ,rebuf , \
        video_chunk_size ,next_video_chunk_sizes , \
        end_of_video ,video_chunk_remain = \
            net_env.get_video_chunk( bit_rate )

        time_stamp += delay  # in ms
        time_stamp += sleep_time  # in ms

        # reward is video quality - rebuffer penalty - smoothness
        reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
                 - REBUF_PENALTY * rebuf \
                 - SMOOTH_PENALTY * np.abs( VIDEO_BIT_RATE[bit_rate] -
                                            VIDEO_BIT_RATE[last_bit_rate] ) / M_IN_K

        r_batch.append( reward )

        last_bit_rate = bit_rate

        # log time_stamp, bit_rate, buffer_size, reward
        log_file.write( str( time_stamp / M_IN_K ) + '\t' +
                        str( VIDEO_BIT_RATE[bit_rate] ) + '\t' +
                        str( buffer_size ) + '\t' +
                        str( rebuf ) + '\t' +
                        str( video_chunk_size ) + '\t' +
                        str( delay ) + '\t' +
                        str( reward ) + '\n' )
        log_file.flush()

        # retrieve previous state
        if len( s_batch ) == 0:
            state = [np.zeros( (args.S_INFO ,args.S_LEN) )]
        else:
            state = np.array( s_batch[-1] ,copy=True )

        # dequeue history record
        state = np.roll( state ,-1 ,axis=1 )

        # this should be args.S_INFO number of terms
        state[0 ,-1] = VIDEO_BIT_RATE[bit_rate] / \
                       float( np.max( VIDEO_BIT_RATE ) )  # last quality
        state[1 ,-1] = buffer_size / args.BUFFER_NORM_FACTOR  # 10 sec
        state[2 ,-1] = float( video_chunk_size ) / \
                       float( delay ) / M_IN_K  # kilo byte / ms
        state[3 ,-1] = float( delay ) / M_IN_K / \
                       args.BUFFER_NORM_FACTOR  # 10 sec
        state[4 ,:args.BITRATE_DIM] = np.array(
            next_video_chunk_sizes ) / M_IN_K / M_IN_K  # mega byte
        state[5 ,-1] = np.minimum( video_chunk_remain ,
                                   args.CHUNK_TIL_VIDEO_END_CAP ) / float( args.CHUNK_TIL_VIDEO_END_CAP )

        action_prob = actor.predict( np.reshape(
            state ,(1 ,args.S_INFO ,args.S_LEN) ) )
        action_cumsum = np.cumsum( action_prob )
        selection = (action_cumsum > np.random.randint(
            1 ,args.RAND_RANGE ) / float( args.RAND_RANGE )).argmax()
        # TODO: Zhengxu: Why compute bitrate this way?
        # selection = action_prob.argmax()
        bit_rate = calculate_from_selection( selection ,last_bit_rate )
        # Note: we need to discretize the probability into 1/args.RAND_RANGE steps,
        # because there is an intrinsic discrepancy in passing single state and batch states

        s_batch.append( state )

        entropy_record.append( a3c.compute_entropy( action_prob[0] ) )

        if end_of_video:
            log_file.write( '\n' )
            log_file.close()

            last_bit_rate = DEFAULT_QUALITY
            bit_rate = DEFAULT_QUALITY  # use the default action here

            del s_batch[:]
            del a_batch[:]
            del r_batch[:]

            # action_vec = np.array( [VIDEO_BIT_RATE[last_bit_rate] ,VIDEO_BIT_RATE[bit_rate] ,selection] )

            action_vec = np.zeros( args.A_DIM )
            action_vec[selection] = 1
            s_batch.append( np.zeros( (args.S_INFO ,args.S_LEN) ) )
            a_batch.append( action_vec )
            entropy_record = []

            video_count += 1

            if video_count >= len( all_file_names ):
                break

            log_path = os.path.join(
                log_output_dir ,
                'log_sim_rl_{}'.format( all_file_names[net_env.trace_idx] ) )
            log_file = open( log_path ,'w' )

    test_dir = log_output_dir
    plot_files = os.listdir( test_dir )

    reward = given_string_mean_reward( plot_files ,test_dir ,str='' )
    reward = round( float( reward ) ,3 )

    return reward

def test(args, test_traces_dir, actor, log_output_dir):
    np.random.seed(args.RANDOM_SEED)

    all_cooked_time, all_cooked_bw, all_file_names = load_traces(
        test_traces_dir)
    # handle the noise and duration variation here
    rtt_test_range = [20, 40, 80, 160, 320]
    rtt_test_result = []

    buffer_test_range = [5000.0, 10000.0, 60000.0, 400000.0, 2000000.0]
    buffer_test_result = []

    payload_test_range = [0.15, 0.35, 0.55, 0.75, 0.95]
    payload_test_result = []

    params_dict = {'buffer_thresh': BUFFER_THRESH ,
                   'drain_buffer_sleep_time': DRAIN_BUFFER_SLEEP_TIME,
                   'packet_payload_portion': PACKET_PAYLOAD_PORTION}

    for param_i in payload_test_range:
        params_dict['packet_payload_portion'] = param_i
        reward = test_on_env_params(args, params_dict, all_cooked_time, all_cooked_bw, all_file_names, actor, log_output_dir)
        payload_test_result.append(reward)

    # rl_mean_reward = {'buffer-5': rtt_test_result[0] ,
    #                   'buffer-10': rtt_test_result[1] ,
    #                   'buffer-60': rtt_test_result[2] ,
    #                   'buffer-400': rtt_test_result[3] ,
    #                   'buffer-2000': rtt_test_result[4]}
    #
    # mpc_mean_reward = {'buffer-5': 0.322 ,
    #                    'buffer-10': 0.599 ,
    #                    'buffer-60': 0.822 ,
    #                    'buffer-400': 0.822 ,
    #                    'buffer-2000': 0.822}

    rl_mean_reward = {'payload-0.6': payload_test_result[0] ,
                      'payload-0.8': payload_test_result[1] ,
                      'payload-0.95': payload_test_result[2]}

    # mpc_mean_reward = {'payload-0.15': -796.195 ,
    #                    'payload-0.35': -95.436 ,
    #                    'payload-0.55': -27.974 ,
    #                    'payload-0.75': -13.011 ,
    #                    'payload-0.95': -10.224}

    # mpc_mean_reward = {'payload-0.15': -413.933 ,
    #                    'payload-0.35': -164.961 ,
    #                    'payload-0.55': -92.706 ,
    #                    'payload-0.75': -57.689 ,
    #                     'payload-0.95': -34.273}
    mpc_mean_reward = {'payload-0.6': -80.17, 'payload-0.8': -46.31, 'payload-0.95': -31.48}

    print( rl_mean_reward ,"-----rl_mean_reward-----" )
    d3 = {key: mpc_mean_reward[key] - rl_mean_reward.get( key ,0 ) for key in rl_mean_reward}

    rl_path = os.path.join( args.summary_dir ,'RL_MPC_log' )
    rl_file = open( rl_path ,'a' ,1 )
    rl_file.write(str( d3 ) + '\n' )
    print( d3 ,"-----mpc - rl-----" )


def given_string_mean_reward(plot_files ,test_dir ,str):
    matching = [s for s in plot_files if str in s]
    count=0
    reward_all = []
    for log_file in matching:
        count+=1
        #print(log_file)
        with open( test_dir +'/'+ log_file ,'r' ) as f:
            raw_reward_all = []
            for line in f:
                parse = line.split()
                if len( parse ) <= 1:
                    break
                raw_reward_all.append( float( parse[6] ) )
            reward_all.append( np.sum( raw_reward_all[1:48] ) / 48 )

    mean = np.mean( reward_all )
    #error_bar = np.std( each_reward )
    mean = round(float(mean), 2)
    #error_bar = round(float(error_bar), 2)

    return mean


def testing(args, epoch, actor, log_file, trace_dir, test_log_folder):
    # clean up the test results folder
    os.system('rm -r ' + test_log_folder)
    os.makedirs(test_log_folder, exist_ok=True)

    # run test script
    test(args, trace_dir, actor, test_log_folder)

    # append test performance to the log
    rewards = []
    test_log_files = os.listdir(test_log_folder)
    print(len(test_log_files))
    for test_log_file in test_log_files:
        reward = []
        with open(os.path.join(test_log_folder, test_log_file), 'r') as f:
            for line in f:
                parse = line.split()
                try:
                    reward.append(float(parse[-1]))
                except IndexError:
                    break
        rewards.append(np.sum(reward[1:]))

    rewards = np.array(rewards)

    rewards_min = np.min(rewards)
    rewards_5per = np.percentile(rewards, 5)
    rewards_mean = np.mean(rewards)
    rewards_median = np.percentile(rewards, 50)
    rewards_95per = np.percentile(rewards, 95)
    rewards_max = np.max(rewards)

    log_file.write(str(epoch) + '\t' +
                   str(rewards_min) + '\t' +
                   str(rewards_5per) + '\t' +
                   str(rewards_mean) + '\t' +
                   str(rewards_median) + '\t' +
                   str(rewards_95per) + '\t' +
                   str(rewards_max) + '\n')
    log_file.flush()
    return rewards_mean


# , train_trace_dir, val_trace_dir, test_trace_dir, noise, duration):
def central_agent(args, net_params_queues, exp_queues):
    # Visdom Logs
    testing_epochs = []
    training_losses = []
    testing_mean_rewards = []
    average_rewards = []
    average_entropies = []

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.logging.set_verbosity(tf.logging.INFO)

    assert len(net_params_queues) == args.NUM_AGENTS
    assert len(exp_queues) == args.NUM_AGENTS

    logging.basicConfig(filename=os.path.join(args.summary_dir, 'log_central'),
                        filemode='w', level=logging.INFO)

    with tf.Session() as sess, \
            open(os.path.join(args.summary_dir, 'log_test'), 'w', 1) as test_log_file, \
            open(os.path.join(args.summary_dir, 'log_train'), 'w', 1) as log_central_file, \
            open(os.path.join(args.summary_dir, 'log_val'), 'w', 1) as val_log_file, \
            open(os.path.join(args.summary_dir, 'log_train_e2e'), 'w', 1) as train_e2e_log_file:
        log_writer = csv.writer(log_central_file, delimiter='\t')
        log_writer.writerow(['epoch', 'loss', 'avg_reward', 'avg_entropy'])
        test_log_file.write("\t".join(
            ['epoch', 'rewards_min', 'rewards_5per', 'rewards_mean',
             'rewards_median', 'rewards_95per', 'rewards_max\n']))
        val_log_file.write("\t".join(
            ['epoch', 'rewards_min', 'rewards_5per', 'rewards_mean',
             'rewards_median', 'rewards_95per', 'rewards_max\n']))
        train_e2e_log_file.write("\t".join(
            ['epoch', 'rewards_min', 'rewards_5per', 'rewards_mean',
             'rewards_median', 'rewards_95per', 'rewards_max\n']))

        actor = a3c.ActorNetwork(sess,
                                 state_dim=[args.S_INFO, args.S_LEN],
                                 action_dim=args.A_DIM,
                                 bitrate_dim=args.BITRATE_DIM)
                                 # learning_rate=args.ACTOR_LR_RATE)
        critic = a3c.CriticNetwork(sess,
                                   state_dim=[args.S_INFO, args.S_LEN],
                                   learning_rate=args.CRITIC_LR_RATE,
                                   bitrate_dim=args.BITRATE_DIM)

        logging.info('actor and critic initialized')
        summary_ops, summary_vars = a3c.build_summaries()

        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(
            args.summary_dir, sess.graph)  # training monitor
        saver = tf.train.Saver(max_to_keep=15)  # save neural net parameters

        # restore neural net parameters
        if args.nn_model is not None:  # nn_model is the path to file
            saver.restore(sess, args.nn_model)
            print("Model restored.")

        epoch = 0

        # assemble experiences from agents, compute the gradients
        max_avg_reward = None
        while True:
            start_t = time.time()
            # synchronize the network parameters of work agent
            actor_net_params = actor.get_network_params()
            critic_net_params = critic.get_network_params()
            for i in range(args.NUM_AGENTS):
                net_params_queues[i].put([actor_net_params, critic_net_params])
                # Note: this is synchronous version of the parallel training,
                # which is easier to understand and probe. The framework can be
                # fairly easily modified to support asynchronous training.
                # Some practices of asynchronous training (lock-free SGD at
                # its core) are nicely explained in the following two papers:
                # https://arxiv.org/abs/1602.01783
                # https://arxiv.org/abs/1106.5730

            # record average reward and td loss change
            # in the experiences from the agents
            total_batch_len = 0.0
            total_reward = 0.0
            total_td_loss = 0.0
            total_entropy = 0.0
            total_agents = 0.0

            # assemble experiences from the agents
            actor_gradient_batch = []
            critic_gradient_batch = []

            # linear entropy weight decay(paper sec4.4)
            entropy_weight = entropy_weight_decay_func(epoch)
            current_learning_rate = learning_rate_decay_func(epoch)

            for i in range(args.NUM_AGENTS):
                # (47, 6, 8) (47,6) (47,)
                s_batch, a_batch, r_batch, terminal, info = exp_queues[i].get()
                #print(np.array(a_batch).shape)

                actor_gradient, critic_gradient, td_batch = \
                    a3c.compute_gradients(
                        s_batch=np.stack(s_batch, axis=0),
                        a_batch=np.vstack(a_batch),
                        r_batch=np.vstack(r_batch),
                        terminal=terminal, actor=actor, critic=critic,
                        entropy_weight=entropy_weight)

                actor_gradient_batch.append(actor_gradient)
                critic_gradient_batch.append(critic_gradient)

                total_reward += np.sum(r_batch)
                total_td_loss += np.sum(td_batch)
                total_batch_len += len(r_batch)
                total_agents += 1.0
                total_entropy += np.sum(info['entropy'])

            # compute aggregated gradient
            assert args.NUM_AGENTS == len(actor_gradient_batch)
            assert len(actor_gradient_batch) == len(critic_gradient_batch)
            # assembled_actor_gradient = actor_gradient_batch[0]
            # assembled_critic_gradient = critic_gradient_batch[0]
            # for i in range(len(actor_gradient_batch) - 1):
            #     for j in range(len(assembled_actor_gradient)):
            #             assembled_actor_gradient[j] += actor_gradient_batch[i][j]
            #             assembled_critic_gradient[j] += critic_gradient_batch[i][j]
            # actor.apply_gradients(assembled_actor_gradient)
            # critic.apply_gradients(assembled_critic_gradient)
            for i in range(len(actor_gradient_batch)):
                actor.apply_gradients(actor_gradient_batch[i], current_learning_rate)
                critic.apply_gradients(critic_gradient_batch[i])

            # log training information
            epoch += 1
            avg_reward = total_reward / total_agents
            avg_td_loss = total_td_loss / total_batch_len
            avg_entropy = total_entropy / total_batch_len

            logging.info('Epoch: ' + str(epoch) +
                         ' TD_loss: ' + str(avg_td_loss) +
                         ' Avg_reward: ' + str(avg_reward) +
                         ' Avg_entropy: ' + str(avg_entropy))
            log_writer.writerow([epoch, avg_td_loss, avg_reward, avg_entropy])

            summary_str = sess.run(summary_ops, feed_dict={
                summary_vars[0]: avg_td_loss,
                summary_vars[1]: avg_reward,
                summary_vars[2]: avg_entropy
            })

            writer.add_summary(summary_str, epoch)
            writer.flush()

            if epoch % MODEL_SAVE_INTERVAL == 0:
                # Visdom log and plot
                test_mean_reward = testing(
                    args, epoch, actor, val_log_file, args.val_trace_dir,
                    os.path.join(args.summary_dir, 'test_results'))
                testing_epochs.append(epoch)
                testing_mean_rewards.append(test_mean_reward)
                average_rewards.append(np.sum(avg_reward))
                average_entropies.append(avg_entropy)

                suffix = args.start_time
                if args.description is not None:
                    suffix = args.description
                trace = dict(x=testing_epochs, y=testing_mean_rewards, mode="markers+lines", type='custom',
                             marker={'color': PLOT_COLOR,
                                     'symbol': 104, 'size': "5"},
                             text=["one", "two", "three"], name='1st Trace')
                layout = dict(title="Pensieve_Val_Reward " + suffix,
                              xaxis={'title': 'Epoch'},
                              yaxis={'title': 'Mean Reward'})
                vis._send(
                    {'data': [trace], 'layout': layout, 'win': 'Pensieve_val_mean_reward_' + args.start_time})
                trace = dict(x=testing_epochs, y=average_rewards, mode="markers+lines", type='custom',
                             marker={'color': PLOT_COLOR,
                                     'symbol': 104, 'size': "5"},
                             text=["one", "two", "three"], name='1st Trace')
                layout = dict(title="Pensieve_Training_Reward " + suffix,
                              xaxis={'title': 'Epoch'},
                              yaxis={'title': 'Mean Reward'})
                vis._send(
                    {'data': [trace], 'layout': layout, 'win': 'Pensieve_training_mean_reward_' + args.start_time})
                trace = dict(x=testing_epochs, y=average_entropies, mode="markers+lines", type='custom',
                             marker={'color': PLOT_COLOR,
                                     'symbol': 104, 'size': "5"},
                             text=["one", "two", "three"], name='1st Trace')
                layout = dict(title="Pensieve_Training_Mean Entropy " + suffix,
                              xaxis={'title': 'Epoch'},
                              yaxis={'title': 'Mean Entropy'})
                vis._send(
                    {'data': [trace], 'layout': layout, 'win': 'Pensieve_training_mean_entropy_' + args.start_time})

                # avg_val_reward = testing(
                #     args, epoch, actor, val_log_file, args.val_trace_dir,
                #     os.path.join(args.summary_dir, 'test_results'),
                #     args.noise, args.duration)
                if max_avg_reward is None or (test_mean_reward > max_avg_reward):
                    max_avg_reward = test_mean_reward
                    # Save the neural net parameters to disk.
                    save_path = saver.save(
                        sess,
                        os.path.join(args.summary_dir+"/model_saved/", f"nn_model_ep_{epoch}.ckpt"))
                    logging.info("Model saved in file: " + save_path)

            end_t = time.time()
            print(f'epoch{epoch-1}: {end_t - start_t}s')


def agent(args, param_list, agent_id, all_cooked_time, all_cooked_bw, all_file_names,
          net_params_queue, exp_queue):

    net_env = env.Environment(buffer_thresh=BUFFER_THRESH,
                              drain_buffer_sleep_time=DRAIN_BUFFER_SLEEP_TIME,
                              packet_payload_portion=PACKET_PAYLOAD_PORTION,
                              link_rtt=LINK_RTT,
                              all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw,
                              all_file_names=all_file_names,
                              random_seed=agent_id)

    with tf.Session() as sess, open(os.path.join(
            args.summary_dir, f'log_agent_{agent_id}'), 'w') as log_file:

        # log_file.write('\t'.join(['time_stamp', 'bit_rate', 'buffer_size',
        #                'rebuffer', 'video_chunk_size', 'delay', 'reward',
        #                'epoch', 'trace_idx', 'mahimahi_ptr'])+'\n')
        actor = a3c.ActorNetwork(sess,
                                 state_dim=[args.S_INFO, args.S_LEN],
                                 action_dim=args.A_DIM,
                                 bitrate_dim=args.BITRATE_DIM)
                                 # learning_rate=args.ACTOR_LR_RATE)
        critic = a3c.CriticNetwork(sess,
                                   state_dim=[args.S_INFO, args.S_LEN],
                                   learning_rate=args.CRITIC_LR_RATE,
                                   bitrate_dim=args.BITRATE_DIM)

        # initial synchronization of the network parameters from the coordinator
        actor_net_params, critic_net_params = net_params_queue.get()
        actor.set_network_params(actor_net_params)
        critic.set_network_params(critic_net_params)

        last_bit_rate = DEFAULT_QUALITY
        selection = 0
        bit_rate = DEFAULT_QUALITY

        #action_vec = np.array( [VIDEO_BIT_RATE[last_bit_rate] ,VIDEO_BIT_RATE[bit_rate] ,selection] )
        action_vec = np.zeros( args.A_DIM )
        action_vec[selection] = 1

        s_batch = [np.zeros((args.S_INFO, args.S_LEN))]
        a_batch = [action_vec]
        r_batch = []
        entropy_record = []

        time_stamp = 0
        epoch = 0
        while True:  # experience video streaming forever
            # the action is from the last decision
            # this is to make the framework similar to the real
            delay, sleep_time, buffer_size, rebuf, \
                video_chunk_size, next_video_chunk_sizes, \
                end_of_video, video_chunk_remain = \
                net_env.get_video_chunk(bit_rate)

            time_stamp += delay  # in ms
            time_stamp += sleep_time  # in ms

            # -- linear reward --
            # reward is video quality - rebuffer penalty - smoothness
            reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
                - REBUF_PENALTY * rebuf \
                - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                          VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K


            r_batch.append(reward)
            last_bit_rate = bit_rate

            # retrieve previous state
            if len(s_batch) == 0:
                state = [np.zeros((args.S_INFO, args.S_LEN))]
            else:
                state = np.array(s_batch[-1], copy=True)

            # dequeue history record
            state = np.roll(state, -1, axis=1)

            # this should be args.S_INFO number of terms
            state[0, -1] = VIDEO_BIT_RATE[bit_rate] / \
                float(np.max(VIDEO_BIT_RATE))  # last quality
            # state[0 ,-1] = VIDEO_BIT_RATE[bit_rate] + \
            #          float(selection)  # last quality
            # state[0 ,-1] = VIDEO_BIT_RATE[bit_rate] / \
            #                VIDEO_BIT_RATE[last_bit_rate]  # last quality
            state[1, -1] = buffer_size / args.BUFFER_NORM_FACTOR  # 10 sec
            state[2, -1] = float(video_chunk_size) / \
                float(delay) / M_IN_K  # kilo byte / ms
            state[3, -1] = float(delay) / M_IN_K / \
                args.BUFFER_NORM_FACTOR  # 10 sec
            state[4, :args.BITRATE_DIM] = np.array(
                next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
            state[5, -1] = np.minimum(video_chunk_remain,
                                      args.CHUNK_TIL_VIDEO_END_CAP) / float(args.CHUNK_TIL_VIDEO_END_CAP)

            # compute action probability vector
            action_prob = actor.predict(np.reshape(
                state, (1, args.S_INFO, args.S_LEN)))
            #print(action_prob, "action prob")
            action_cumsum = np.cumsum(action_prob)
            selection = (action_cumsum > np.random.randint(
                1, args.RAND_RANGE) / float(args.RAND_RANGE)).argmax()
            bit_rate = calculate_from_selection( selection, last_bit_rate )
            #print(bit_rate, "bitrate")
            # Note: we need to discretize the probability into 1/args.RAND_RANGE steps,
            # because there is an intrinsic discrepancy in passing single state and batch states

            entropy_record.append(a3c.compute_entropy(action_prob[0]))

            # log time_stamp, bit_rate, buffer_size, reward
            # log_file.write(str(time_stamp) + '\t' +
            #                str(VIDEO_BIT_RATE[bit_rate]) + '\t' +
            #                str(buffer_size) + '\t' +
            #                str(rebuf) + '\t' +
            #                str(video_chunk_size) + '\t' +
            #                str(delay) + '\t' +
            #                str(reward) + '\t' +
            #                str(epoch) + '\t' +
            #                str(net_env.trace_idx) + '\t' +
            #                str(net_env.mahimahi_ptr)+'\n')
            # log_file.flush()

            # report experience to the coordinator
            if len(r_batch) >= args.TRAIN_SEQ_LEN or end_of_video:
                exp_queue.put([s_batch[1:],  # ignore the first chuck
                               a_batch[1:],  # since we don't have the
                               r_batch[1:],  # control over it
                               end_of_video,
                               {'entropy': entropy_record}])

                # synchronize the network parameters from the coordinator
                actor_net_params, critic_net_params = net_params_queue.get()
                actor.set_network_params(actor_net_params)
                critic.set_network_params(critic_net_params)

                del s_batch[:]
                del a_batch[:]
                del r_batch[:]
                del entropy_record[:]

                # so that in the log we know where video ends
                log_file.write('\n')

            # store the state and action into batches
            if end_of_video:
                last_bit_rate = DEFAULT_QUALITY
                bit_rate = DEFAULT_QUALITY  # use the default action here

                #action_vec = np.array( [VIDEO_BIT_RATE[last_bit_rate] ,VIDEO_BIT_RATE[bit_rate] ,selection] )
                action_vec = np.zeros( args.A_DIM )
                action_vec[selection] = 1
                s_batch.append(np.zeros((args.S_INFO, args.S_LEN)))
                a_batch.append(action_vec)
                epoch += 1

                # if epoch > 1 and epoch % UPDATE_ENV_INTERVAL == 0:
                #     net_env.packet_payload_portion = param_for_epoch( epoch, param_list )
                #     print( net_env.packet_payload_portion ,"-----net_env.packet_payload_portion" )

            else:
                s_batch.append(state)
                action_vec = np.zeros( args.A_DIM )
                action_vec[selection] = 1
                #print(action_vec)
                a_batch.append(action_vec)


def param_for_epoch(epoch ,param_list):
    """
    epoch is epoch 1, 2, 3 ,etc.
    rtts: list of rtt values
    returns a value from the list for the epoch
    1 -> 123
    200 -> 186
    201 -> 186
    """
    index = epoch // UPDATE_ENV_INTERVAL
    param = param_list[index]
    return param

def main(args):

    start_time = datetime.now()
    start_time_string = start_time.strftime("%Y%m%d_%H%M%S")
    args.start_time = start_time_string

    np.random.seed(args.RANDOM_SEED)
    #assert len(VIDEO_BIT_RATE) == args.A_DIM

    #rtt_list = np.random.randint(LINK_RTT_MIN, LINK_RTT_MAX, size=100)
    payload_list = np.round( np.random.uniform( 0.1 ,0.99 ,size=50 ) ,2 )

    param_list = payload_list
    # create result directory
    if not os.path.exists(args.summary_dir):
        os.makedirs(args.summary_dir)

    config.log_config(args)

    # inter-process communication queues
    net_params_queues = []
    exp_queues = []
    for i in range(args.NUM_AGENTS):
        net_params_queues.append(mp.Queue(1))
        exp_queues.append(mp.Queue(1))

    # create a coordinator and multiple agent processes
    # (note: threading is not desirable due to python GIL)
    coordinator = mp.Process(target=central_agent,
                             args=(args, net_params_queues, exp_queues))
    coordinator.start()

    all_cooked_time, all_cooked_bw, all_file_names = load_traces(
        args.train_trace_dir)
    agents = []
    for i in range(args.NUM_AGENTS):
        agents.append(mp.Process(target=agent,
                                 args=(args, param_list, i, all_cooked_time, all_cooked_bw,
                                       all_file_names, net_params_queues[i],
                                       exp_queues[i])))
    for i in range(args.NUM_AGENTS):
        agents[i].start()

    # wait unit training is done
    coordinator.join()


if __name__ == '__main__':
    args = config.parse_args()

    main(args)
