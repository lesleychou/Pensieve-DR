import argparse
import csv
# import os
import random
import time as time_module

import numpy as np


def parse_args():
    '''
    Parse arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Generate synthetic data.")
    parser.add_argument("--T_l", type=float, required=True,
                        help='control the prob_stay')
    parser.add_argument("--T_s", type=float, required=True,
                        help='control how long to recompute noise')
    parser.add_argument("--cov", type=float, required=True,
                        help='coefficient used to compute vairance of a state')
    parser.add_argument("--duration", type=float, required=True,
                        help='duration of each synthetic trace in seconds.')
    parser.add_argument("--steps", type=int, default=10,
                        help='number of steps')
    parser.add_argument("--switch-parameter", type=float, required=True,
                        help='switch parameter solved by polynomial solver')
    parser.add_argument("--max-throughput", type=float, default=4.3,
                        help='upper bound of throughput(Mbps)')
    parser.add_argument("--min-throughput", type=float, default=0.2,
                        help='lower bound of throughput(Mbps)')
    parser.add_argument("--output_file", type=str, required=True,
                        help='Output file name.')

    return parser.parse_args()


def transition(state, variance, prob_stay, bitrate_states_low_var,
               transition_probs):
    # variance_switch_prob, sigma_low, sigma_high,
    transition_prob = random.uniform(0, 1)

    # pick next variance first
    # variance_switch = random.uniform(0, 1)
    # next_variance = variance
    # if (variance_switch < variance_switch_prob):
    #     if (next_variance == sigma_low):
    #         next_variance = sigma_high
    #     else:
    #         next_variance = sigma_low

    if transition_prob < prob_stay:  # stay in current state
        return state
    else:  # pick appropriate state!
        # next_state = state
        curr_pos = state
        # first find max distance that you can be from current state
        max_distance = max(curr_pos, len(bitrate_states_low_var)-1-curr_pos)
        # cut the transition probabilities to only have possible number of
        # steps
        curr_transition_probs = transition_probs[0:max_distance]
        trans_sum = sum(curr_transition_probs)
        normalized_trans = [x/trans_sum for x in curr_transition_probs]
        # generate a random number and see which bin it falls in to
        trans_switch_val = random.uniform(0, 1)
        running_sum = 0
        num_switches = -1
        for ind in range(0, len(normalized_trans)):
            # this is the val
            if (trans_switch_val <= (normalized_trans[ind] + running_sum)):
                num_switches = ind
                break
            else:
                running_sum += normalized_trans[ind]

        # now check if there are multiple ways to move this many states away
        switch_up = curr_pos + num_switches
        switch_down = curr_pos - num_switches
        # can go either way
        if (switch_down >= 0 and switch_up <= (len(bitrate_states_low_var)-1)):
            x = random.uniform(0, 1)
            if (x < 0.5):
                return switch_up
            else:
                return switch_down
        elif switch_down >= 0:  # switch down
            return switch_down
        else:  # switch up
            return switch_up


def main():
    args = parse_args()
    T_s = args.T_s
    T_l = args.T_l
    cov = args.cov
    time_length = args.duration
    output_file = args.output_file
    output_writer = csv.writer(open(output_file, 'w', 1), delimiter='\t')
    # get bitrate levels (in Mbps)
    bitrate_states_low_var = []
    curr = args.min_throughput
    for _ in range(0, args.steps):
        bitrate_states_low_var.append(curr)
        curr += ((args.max_throughput - args.min_throughput)/(args.steps - 1))

    # list of transition probabilities
    transition_probs = []
    # assume you can go steps-1 states away (we will normalize this to the
    # actual scenario)
    for z in range(1, args.steps-1):
        transition_probs.append(1/(args.switch_parameter**z))
    # two variance levels
    # sigma_low = 1.0
    # sigma_high = 1.0

    # probability of switching variance levels
    # variance_switch_prob = 0.2

    # probability to stay in same state
    prob_stay = 1 - 1 / T_l

    # takes a state and decides what the next state is

    current_state = random.randint(0, len(bitrate_states_low_var)-1)
    current_variance = cov * bitrate_states_low_var[current_state]
    time = 0
    cnt = 0
    while time < time_length:
        # prints timestamp (in seconds) and throughput (in Mbits/s)
        if cnt <= 0:
            noise = np.random.normal(0, current_variance, 1)[0]
            cnt = T_s
        # TODO: the gaussian val is at least 0.1
        final_bw = bitrate_states_low_var[current_state] + noise
        if final_bw < args.min_throughput:
            final_bw = args.min_throughput
        if final_bw > args.max_throughput:
            final_bw = args.max_throughput
        gaus_val = max( 0.1 ,final_bw )
        #gaus_val = max(0.1, bitrate_states_low_var[current_state] + noise)
        output_writer.writerow([time, gaus_val])
        cnt -= 1
        next_val = transition(current_state, current_variance, prob_stay,
                              bitrate_states_low_var, transition_probs)
        # print(next_vals)
        if current_state != next_val:
            cnt = 0
        current_state = next_val
        current_variance = cov * bitrate_states_low_var[current_state]
        time += 1
    # print('e2e running used {}s'.format(time_module.time()-start))


if __name__ == "__main__":
    main()