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
    parser.add_argument("--max-throughput", type=float, default=4.3,
                        help='upper bound of throughput(Mbps)')
    parser.add_argument("--min-throughput", type=float, default=0.2,
                        help='lower bound of throughput(Mbps)')
    parser.add_argument("--output_file", type=str, required=True,
                        help='Output file name.')

    return parser.parse_args()

def main():
    args = parse_args()
    T_s = args.T_s
    time_length = args.duration
    min_tp = args.min_throughput
    max_tp = args.max_throughput
    output_file = args.output_file
    output_writer = csv.writer(open(output_file, 'w', 1), delimiter='\t')

    round_digit = 1

    time = 0
    cnt = 0
    last_val = round(np.random.uniform(min_tp, max_tp), round_digit)

    while time < time_length:
        if cnt <= 0:
            bw_val = round( np.random.uniform( min_tp ,max_tp ) ,round_digit )
            cnt = np.random.randint(1, T_s+1)
        elif cnt >= 1:
            bw_val = last_val
        else:
            bw_val = round(np.random.uniform(min_tp, max_tp), round_digit)

        cnt -= 1
        output_writer.writerow([time, bw_val])

        last_val = bw_val
        time += 1


if __name__ == "__main__":
    main()