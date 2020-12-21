import argparse
import logging

def parse_args():
    '''
    Parse arguments from the command line.
    '''
    parser = argparse.ArgumentParser("MORL_ABR")
    parser.add_argument('--description', type=str, default=None, help='Optional description of the experiment.')

    # bit_rate, buffer_size, next_chunk_size, bw_measurement (throughput and time), chunk_til_video_end
    parser.add_argument('--S_INFO', type=int, default='6', help='State info shape. Defaults to 6.')
    parser.add_argument('--S_LEN', type=int, default='8', help='How many frames in the past to consider. Defaults to 8.')
    parser.add_argument('--A_DIM', type=int, default='6', help='Action dimension. Defaults to 6.')
    parser.add_argument('--ACTOR_LR_RATE', type=float, default='0.0001', help='Actor learning rate. Defaults to 0.0001.')
    parser.add_argument('--CRITIC_LR_RATE', type=float, default='0.0001', help='Critic learning rate. Defaults to 0.0001.')
    parser.add_argument('--NUM_AGENTS', type=int, default='4', help='Num of worker agents. Defaults to 4.')
    parser.add_argument('--NUM_GAE_STEP', type=int, default='1', help='Num of gae steps. Defaults to 1.')
    parser.add_argument('--ACTOR_FILEPATH', type=str, default=None, help='Actor model file path.')

    # unused now
    parser.add_argument('--TRAIN_SEQ_LEN', type=int, default='100', help='take as a train batch')
    parser.add_argument('--MODEL_SAVE_INTERVAL', type=int, default='100', help='')

    parser.add_argument('--REBUF_PENALTY', type=float, default='4.0', help='use 1 for linear, 4 for HD')
    parser.add_argument('--SMOOTH_PENALTY', type=float, default='1.0', help='')
    parser.add_argument('--DEFAULT_QUALITY', type=int, default='1', help='default video quality without agent')

    parser.add_argument('--RANDOM_SEED', type=int, default='171', help='')
    # unused now
    parser.add_argument('--RAND_RANGE', type=int, default='1000', help='')

    parser.add_argument('--REWARD_SCALE', type=int, default='1', help='')
    parser.add_argument('--SAMPLE_SIZE', type=int, default='24', help='number of preference samples for updating')
    parser.add_argument('--TOTAL_EPOCH', type=int, default='50000', help='total training epoch')
    parser.add_argument('--TEMPERATURE', type=int, default='10', help='Temperature to encourage exploration')
    parser.add_argument('--ENVELOPE_START', type=int, default=1000, help='global step at which to start the envelope calculation')
    parser.add_argument('--bitrate-exp', type=float, default=2.5, help='Applies exponent to bitrate part of reward.')
        
    # for video state
    parser.add_argument('--BUFFER_NORM_FACTOR', type=float, default='10.0', help='')
    parser.add_argument('--CHUNK_TIL_VIDEO_END_CAP', type=float, default='48.0', help='')
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    # helper
    parser.add_argument('--TEST_RESULTS_DIR', type=str, default=None, help='')
    # adjust learning rate
    parser.add_argument('--lr-schedule', type=bool, default=True, help='enable learning rate scheduling')
    parser.add_argument('--max-step', type=int, default=12e5, metavar='MSTEP',
                        help='max number of steps for learning rate scheduling (default 1.15e8)')
    # for the critic sync
    parser.add_argument('--update-target-critic', type=int, default=5e3, metavar='UTC',
                        help='the number of steps to update target critic')
    parser.add_argument('--fixed-weights', dest='fixed_weights', action='store_true', default=False,
                        help='If true, use fixed weights instead of dynamic weights.')
    parser.add_argument('--bitrate-weight', type=float, default='0.5', help='for fixed-weights setting')
    parser.add_argument('--rebuf-weight', type=float, default='0.5', help='for fixed-weights setting')


    parser.set_defaults(fixed_weights=False)
    parser.add_argument('--objective', type=str, default="multi", help='either single or multi')
    
    return parser.parse_args()

def log_config(args):
    '''
    Writes arguments to log. Assumes args.results_dir exists.
    '''
    log_file = args.results_dir + 'config'
    config_logging = logging.getLogger("config")
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    config_logging.setLevel(logging.INFO)
    config_logging.addHandler(file_handler)
    config_logging.addHandler(stream_handler)
    for arg in vars(args):
        config_logging.info(arg + '\t' + str(getattr(args, arg)))

def results_subdir(args):
    name = args.start_time
    if args.description is not None:
        name = name + '_' + args.description
    return name
