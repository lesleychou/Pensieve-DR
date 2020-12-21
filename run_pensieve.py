'''
This is the Pensieve training with random weights
'''

import os
import logging
from datetime import datetime
import random

import numpy as np
import torch.multiprocessing as mp
import torch
import visdom

import env
import load_trace

import src.config as config
from src import reward as mor
# import src.weights_helper as weights_helper
import src.weights_helper as exp_weight
from src.agent import A3C_Pensieve

VIDEO_BIT_RATE = [300, 1200, 2850, 6500, 33000, 165000]  # Kbps
HD_REWARD = [1, 2, 3, 12, 15, 20]
M_IN_K = 1000.0

TRAIN_TRACES = './data/generated_traces/'

IS_CENTRAL = True
NO_CENTRAL = False
CRITIC_MODEL = 'critic.pt'
ACTOR_MODEL = 'actor.pt'

testing_epochs = []
testing_mean_rewards = []

vis = visdom.Visdom()
assert vis.check_connection()
PLOT_COLOR = 'blue'

def testing(args, epoch, actor_model, log_file):

    test_results_dir = args.results_dir + "test_results/" + str(epoch) + "/"
    if not os.path.exists(test_results_dir):
        os.makedirs(test_results_dir)

    command = "python pensieve_test.py " \
              "--TEST_RESULTS_DIR={test_results_dir} " \
              "--ACTOR_FILEPATH={actor_model} " \
              "--RANDOM_SEED={args.RANDOM_SEED} " \
              "--A_DIM={args.A_DIM} " \
              "--S_LEN={args.S_LEN} " \
              "--S_INFO={args.S_INFO} " \
              "--DEFAULT_QUALITY={args.DEFAULT_QUALITY} " \
              "--REBUF_PENALTY={args.REBUF_PENALTY} " .format(
                  test_results_dir=test_results_dir,
                  actor_model=actor_model,
                  args=args)
    if args.disable_cuda:
        command = command + " --disable-cuda"
    if args.fixed_weights:
        command = command + " --fixed-weights"
    os.system(command)

    # append test performance to the log
    rewards = []
    rebuf_delays = []
    bitrates = []


    test_log_files = os.listdir(test_results_dir)
    for test_log_file in test_log_files:
        reward = []
        rebuf_delay = []
        bitrate = []

        with open(test_results_dir + test_log_file, 'r') as f:
            for line in f:
                parse = line.split()
                try:
                    rebuf_delay.append((float(parse[3])))
                    reward.append(float(parse[6]))
                    bitrate.append((float(parse[1])))
                except IndexError:
                    break
        rewards.append(sum(reward[1:]))
        rebuf_delays.append(sum(rebuf_delay[1:]))
        bitrates.append(sum(bitrate[1:]))

    rewards = np.array(rewards)

    rewards_min = np.min(rewards)
    rewards_5per = np.percentile(rewards, 5)
    rewards_mean = np.mean(rewards)
    rewards_median = np.percentile(rewards, 50)
    rewards_95per = np.percentile(rewards, 95)
    rewards_max = np.max(rewards)
    rebuf_mean = np.mean(rebuf_delays)
    bitrate_mean = np.mean(bitrates)

    log_file.write(str(epoch) + '\t' +
                   str(rewards_min) + '\t' +
                   str(rewards_5per) + '\t' +
                   str(rewards_mean) + '\t' +
                   str(rewards_median) + '\t' +
                   str(rewards_95per) + '\t' +
                   str(rewards_max) + '\t' +
                   str(rebuf_mean) + '\t' +
                   str(bitrate_mean) + '\n')
    log_file.flush()

    testing_epochs.append(epoch)
    testing_mean_rewards.append(rewards_mean)

    suffix = args.start_time
    if args.description is not None:
        suffix = args.description

    trace = dict(x=testing_epochs, y=testing_mean_rewards, mode="markers+lines", type='custom',
                 marker={'color': PLOT_COLOR, 'symbol': 104, 'size': "5"},
                 text=["one", "two", "three"], name='1st Trace')
    layout = dict(title="Pensieve - Testing - Mean Reward " + suffix,
                  xaxis={'title': 'Epoch'},
                  yaxis={'title': 'Mean Reward'})

    vis._send({'data': [trace],
               'layout': layout,
               'win': 'pensieve_testing_mean_reward_' + args.start_time})

def central_agent(args, net_params_queues, exp_queues):
    torch.set_num_threads(1)
    actor_filepath = args.results_dir + ACTOR_MODEL
    print( "Actor Filepath: " + actor_filepath )
    critic_filepath = args.results_dir + CRITIC_MODEL
    print( "Critic Filepath: " + critic_filepath )


    timenow = datetime.now()
    assert len(net_params_queues) == args.NUM_AGENTS
    assert len(exp_queues) == args.NUM_AGENTS

    logging.basicConfig(filename=args.results_dir + 'log_central',
                        filemode='w',
                        level=logging.INFO)

    net = A3C_Pensieve(IS_CENTRAL, [args.S_INFO, args.S_LEN], args.A_DIM, args.ACTOR_LR_RATE, args.CRITIC_LR_RATE)

    if os.path.exists( actor_filepath ):
        print("Model already exists!")
        net.actorNetwork.load_state_dict( torch.load( actor_filepath ) )
        net.criticNetwork.load_state_dict( torch.load( critic_filepath ) )

    test_log_file = open(args.results_dir + 'log_test', 'w')

    # Visdom arrays
    training_losses = []    
    average_rewards = []
    average_entropies = []

    epochs = []
    
    for epoch in range(args.TOTAL_EPOCH):
        # synchronize the network parameters of work agent
        actor_net_params = net.getActorParam()
        # critic_net_params=net.getCriticParam()
        for i in range(args.NUM_AGENTS):
            # net_params_queues[i].put([actor_net_params,critic_net_params])
            net_params_queues[i].put(actor_net_params)
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
        total_loss = 0.0

        # assemble experiences from the agents
        actor_gradient_batch = []
        critic_gradient_batch = []

        for i in range(args.NUM_AGENTS):
            s_batch, a_batch, r_batch, terminal, info = exp_queues[i].get()

            loss = net.getNetworkGradient(s_batch, a_batch, r_batch, terminal=terminal)

            total_reward += np.sum(r_batch)
            total_batch_len += len(r_batch)
            total_agents += 1.0
            total_entropy += np.sum(info['entropy'])
            total_loss += loss

        # log training information
        net.updateNetwork()

        avg_reward = total_reward / total_agents
        avg_entropy = total_entropy / total_batch_len
        avg_loss = total_loss / total_agents
        
        logging.info('Epoch: ' + str(epoch) +
                     ' Avg_reward: ' + str(avg_reward) +
                     ' Avg_entropy: ' + str(avg_entropy) +
                     ' Avg_loss: ' + str(avg_loss))


        if (epoch + 1) % args.MODEL_SAVE_INTERVAL == 0:
            # Save the neural net parameters to disk.
            print("\nTrain ep:" + str(epoch + 1) + ",time use :" + str((datetime.now() - timenow).seconds) + "s\n")
            timenow = datetime.now()
            torch.save(net.actorNetwork.state_dict(), args.results_dir + "/actor.pt")
            torch.save(net.criticNetwork.state_dict(), args.results_dir + "/critic.pt")
            actor = args.results_dir + "actor.pt"
            testing(args, epoch + 1, actor, test_log_file)

            epochs.append(epoch + 1)
            training_losses.append(avg_loss)
            average_rewards.append(avg_reward)
            average_entropies.append(avg_entropy)

            suffix = args.start_time
            if args.description is not None:
                suffix = args.description
            
            trace = dict(x=epochs, y=average_rewards, mode="markers+lines", type='custom',
                         marker={'color': PLOT_COLOR, 'symbol': 104, 'size': "5"},
                         text=["one", "two", "three"], name='1st Trace')
            layout = dict(title="Pensieve - Training - Mean Reward " + suffix,
                          xaxis={'title': 'Epoch'},
                          yaxis={'title': 'Mean Reward'})
            vis._send({'data': [trace], 'layout': layout, 'win': 'pensieve_training_mean_reward_' + args.start_time})
            trace = dict(x=epochs, y=average_entropies, mode="markers+lines", type='custom',
                         marker={'color': PLOT_COLOR, 'symbol': 104, 'size': "5"},
                         text=["one", "two", "three"], name='1st Trace')
            layout = dict(title="Pensieve - Training - Mean Entropy " + suffix,
                          xaxis={'title': 'Epoch'},
                          yaxis={'title': 'Mean Reward'})
            vis._send({'data': [trace], 'layout': layout, 'win': 'pensieve_training_mean_entropy_' + args.start_time})

            trace = dict(x=epochs, y=training_losses, mode="markers+lines", type='custom',
                         marker={'color': PLOT_COLOR, 'symbol': 104, 'size': "5"},
                         text=["one", "two", "three"], name='1st Trace')
            layout = dict(title="Pensieve - Training - Loss " + suffix,
                          xaxis={'title': 'Epoch'},
                          yaxis={'title': 'Mean Reward'})
            vis._send({'data': [trace], 'layout': layout, 'win': 'pensieve_training_loss_' + args.start_time})
            

def agent(args, agent_id, all_cooked_time, all_cooked_bw, net_params_queue, exp_queue):
    torch.set_num_threads(1)

    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw,
                              random_seed=agent_id)

    with open(args.results_dir + 'log_agent_' + str(agent_id), 'w') as log_file:

        net = A3C_Pensieve(NO_CENTRAL, [args.S_INFO, args.S_LEN], args.A_DIM, args.ACTOR_LR_RATE, args.CRITIC_LR_RATE)

        # initial synchronization of the network parameters from the coordinator

        time_stamp = 0
        start_of_video = True

        for epoch in range(args.TOTAL_EPOCH):
            actor_net_params = net_params_queue.get()
            net.hardUpdateActorNetwork(actor_net_params)
            bit_rate = args.DEFAULT_QUALITY
            s_batch = []
            a_batch = []
            r_batch = []
            entropy_record = []
            state = torch.zeros((1, args.S_INFO, args.S_LEN)).to(device=args.device)

            # the action is from the last decision
            # this is to make the framework similar to the real
            delay, sleep_time, buffer_size, rebuf, \
            video_chunk_size, next_video_chunk_sizes, \
            end_of_video, video_chunk_remain = \
                net_env.get_video_chunk(bit_rate)

            time_stamp += delay  # in ms
            time_stamp += sleep_time  # in ms

            while not end_of_video and len(s_batch) < args.TRAIN_SEQ_LEN:
                last_bit_rate = bit_rate

                state = state.clone().detach()

                state = torch.roll(state, -1, dims=-1)

                state[0, 0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))  # last quality
                state[0, 1, -1] = buffer_size / args.BUFFER_NORM_FACTOR  # 10 sec
                state[0, 2, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
                state[0, 3, -1] = float(delay) / M_IN_K / args.BUFFER_NORM_FACTOR  # 10 sec
                state[0, 4, :args.A_DIM] = torch.tensor(next_video_chunk_sizes).to(device=args.device) / M_IN_K / M_IN_K  # mega byte
                state[0, 5, -1] = min(video_chunk_remain, args.CHUNK_TIL_VIDEO_END_CAP) / float(args.CHUNK_TIL_VIDEO_END_CAP)

                bit_rate, entropy = net.select_action(state)

                # Note: we need to discretize the probability into 1/args.RAND_RANGE steps,
                # because there is an intrinsic discrepancy in passing single state and batch states

                delay, sleep_time, buffer_size, rebuf, \
                video_chunk_size, next_video_chunk_sizes, \
                end_of_video, video_chunk_remain = \
                    net_env.get_video_chunk(bit_rate)

                reward = mor.linear_reward(args,bit_rate,rebuf,last_bit_rate)

                s_batch.append(state)
                a_batch.append(bit_rate)
                r_batch.append(reward)
                entropy_record.append(entropy)

                # log time_stamp, bit_rate, buffer_size, reward
                log_file.write(str(time_stamp) + '\t' +
                               str(VIDEO_BIT_RATE[bit_rate]) + '\t' +
                               str(buffer_size) + '\t' +
                               str(rebuf) + '\t' +
                               str(video_chunk_size) + '\t' +
                               str(delay) + '\t' +
                               str(reward) + '\n')
                log_file.flush()

            exp_queue.put([s_batch,  # ignore the first chuck
                           a_batch,  # since we don't have the
                           r_batch,  # control over it
                           end_of_video,
                           {'entropy': entropy_record}])

            if end_of_video:
                # This assumes the end of the video is the end of the epoch.
                # But that is no longer true.
                log_file.write('\n')  # so that in the log we know where video ends


def main(args):
    time = datetime.now()

    os.environ['PYTHONHASHSEED'] = str(args.RANDOM_SEED)
    random.seed(args.RANDOM_SEED)    
    np.random.seed(args.RANDOM_SEED)
    torch.manual_seed(args.RANDOM_SEED)

    assert len(VIDEO_BIT_RATE) == args.A_DIM

    # create result directory
    args.start_time = time.strftime("%Y%m%d_%H%M%S")
    subdir = config.results_subdir(args)
    #results_dir = './results/pensieve/' + subdir + '/'
    results_dir = './results/pensieve/retrain/'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    args.results_dir = results_dir

    config.log_config(args)
    
    # Need to set start method to spawn for multiprocessing in Pytorch.
    # https://stackoverflow.com/questions/48822463/how-to-use-pytorch-multiprocessing
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
        
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

    all_cooked_time, all_cooked_bw, _ = load_trace.load_trace(TRAIN_TRACES)
    agents = []
    for i in range(args.NUM_AGENTS):
        agents.append(mp.Process(target=agent,
                                 args=(args, i, all_cooked_time, all_cooked_bw,
                                       net_params_queues[i],
                                       exp_queues[i])))
    for i in range(args.NUM_AGENTS):
        agents[i].start()

    # wait unit training is done
    coordinator.join()
    for i in range(args.NUM_AGENTS):
        agents[i].join()

    print(str(datetime.now() - time))


if __name__ == '__main__':
    args = config.parse_args()
    args.device = None
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')
    main(args)
