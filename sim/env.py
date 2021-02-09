import numpy as np
from utils.constants import (B_IN_MB, S_INFO, S_LEN, DEFAULT_QUALITY, M_IN_K,
                                MILLISECONDS_IN_SECOND, VIDEO_BIT_RATE,
                                VIDEO_CHUNK_LEN, REBUF_PENALTY, SMOOTH_PENALTY,
                                BITS_IN_BYTE, VIDEO_CHUNK_LEN, BITRATE_LEVELS,TOTAL_VIDEO_CHUNK,
                                PACKET_SIZE,NOISE_LOW, NOISE_HIGH, BUFFER_NORM_FACTOR,CHUNK_TIL_VIDEO_END_CAP)


RANDOM_SEED = 42

BUFFER_THRESH = 60.0 * MILLISECONDS_IN_SECOND  # millisec, max buffer limit
DRAIN_BUFFER_SLEEP_TIME = 500.0  # millisec
PACKET_PAYLOAD_PORTION = 0.95
LINK_RTT = 80  # millisec


VIDEO_SIZE_FILE = '../data/video_size_6_larger/video_size_'


class Environment:
    def __init__(self, all_cooked_time, all_cooked_bw, all_file_names=None,
                 random_seed=RANDOM_SEED, fixed=False):
        assert len(all_cooked_time) == len(all_cooked_bw)

        np.random.seed(random_seed)
        self.fixed = fixed
        self.all_cooked_time = all_cooked_time
        self.all_cooked_bw = all_cooked_bw
        self.all_file_names = all_file_names

        self.video_chunk_counter = 0
        self.buffer_size = 0

        # pick a random trace file
        self.trace_idx = 0 if fixed else np.random.randint(len(self.all_cooked_time))
        #print(self.all_cooked_time, "------idx")
        self.cooked_time = self.all_cooked_time[self.trace_idx]
        self.cooked_bw = self.all_cooked_bw[self.trace_idx]
        self.mahimahi_start_ptr = 1
        self.mahimahi_ptr = 1 if fixed else np.random.randint(1, len(self.cooked_bw))
        # randomize the start point of the trace
        # note: trace file starts with time 0
        self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

        self.video_size = {}  # in bytes
        for bitrate in range(BITRATE_LEVELS):
            self.video_size[bitrate] = []
            with open(VIDEO_SIZE_FILE + str(bitrate)) as f:
                for line in f:
                    self.video_size[bitrate].append(int(line.split()[0]))

        self.total_video_chunk = len(self.video_size[0]) - 1

        # new init for BO-link
        self.state = np.zeros((1, S_INFO, S_LEN))
        self.last_bitrate = DEFAULT_QUALITY

    # def trace_file_name(self):
    #     trace = self.traces[self.trace_idx]
    #     return trace.filename

    def reset(self, **kwargs):
        """Reset the environment state to default values."""
        self.trace_idx = 0
        self.mahimahi_ptr = 1
        self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]
        self.video_chunk_counter = 0
        self.buffer_size = 0
        self.last_bitrate = DEFAULT_QUALITY
        self.state = np.zeros((1, S_INFO, S_LEN))


    def get_video_chunk(self, bitrate):

        assert bitrate >= 0
        assert bitrate < BITRATE_LEVELS

        video_chunk_size = self.video_size[bitrate][self.video_chunk_counter]
        #print(video_chunk_size, "----video_chunk_size---")

        # use the delivery opportunity in mahimahi
        delay = 0.0  # in ms
        video_chunk_counter_sent = 0  # in bytes

        while True:  # download video chunk over mahimahi
            throughput = self.cooked_bw[self.mahimahi_ptr] \
                         * B_IN_MB / BITS_IN_BYTE  # throughput = bytes per ms
            #print(self.cooked_bw[self.mahimahi_ptr], "bw")
            duration = self.cooked_time[self.mahimahi_ptr] \
                       - self.last_mahimahi_time

            packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION

            if video_chunk_counter_sent + packet_payload > video_chunk_size:
                fractional_time = (video_chunk_size - video_chunk_counter_sent) / \
                                  throughput / PACKET_PAYLOAD_PORTION
                delay += fractional_time
                #print( delay ,"--------fractional_time------" )
                self.last_mahimahi_time += fractional_time
                assert(self.last_mahimahi_time <= self.cooked_time[self.mahimahi_ptr])
                break

            video_chunk_counter_sent += packet_payload
            delay += duration
            #print(delay, "--------fractional_time + duration------")

            self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
            self.mahimahi_ptr += 1

            if self.mahimahi_ptr >= len(self.cooked_bw):
                # loop back in the beginning
                # note: trace file starts with time 0
                self.mahimahi_ptr = 1
                self.last_mahimahi_time = 0


        #print( delay ,"--------before------" )
        delay *= MILLISECONDS_IN_SECOND
        #print( delay ,"--------MILLISECONDS_IN_SECOND------" )

        delay += LINK_RTT
        #print( delay ,"--------RTT------" )

        # add a multiplicative noise to the delay
        if not self.fixed:
            delay *= np.random.uniform(NOISE_LOW, NOISE_HIGH)

        # rebuffer time
        rebuf = np.maximum(delay - self.buffer_size, 0.0)

        # update the buffer
        self.buffer_size = np.maximum(self.buffer_size - delay, 0.0)

        # add in the new chunk
        self.buffer_size += VIDEO_CHUNK_LEN

        # sleep if buffer gets too large
        sleep_time = 0
        if self.buffer_size > BUFFER_THRESH:
            # exceed the buffer limit
            # we need to skip some network bandwidth here
            # but do not add up the delay
            drain_buffer_time = self.buffer_size - BUFFER_THRESH
            sleep_time = np.ceil(drain_buffer_time / DRAIN_BUFFER_SLEEP_TIME) * \
                         DRAIN_BUFFER_SLEEP_TIME
            self.buffer_size -= sleep_time

            while True:
                duration = self.cooked_time[self.mahimahi_ptr] \
                           - self.last_mahimahi_time
                if duration > sleep_time / MILLISECONDS_IN_SECOND:
                    self.last_mahimahi_time += sleep_time / MILLISECONDS_IN_SECOND
                    break
                sleep_time -= duration * MILLISECONDS_IN_SECOND
                self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
                self.mahimahi_ptr += 1

                if self.mahimahi_ptr >= len(self.cooked_bw):
                    # loop back in the beginning
                    # note: trace file starts with time 0
                    self.mahimahi_ptr = 1
                    self.last_mahimahi_time = 0

        # the "last buffer size" return to the controller
        # Note: in old version of dash the lowest buffer is 0.
        # In the new version the buffer always have at least
        # one chunk of video
        return_buffer_size = self.buffer_size

        self.video_chunk_counter += 1
        video_chunk_remain = self.total_video_chunk - self.video_chunk_counter

        end_of_video = False
        if self.video_chunk_counter >= self.total_video_chunk:
            end_of_video = True
            self.buffer_size = 0
            self.video_chunk_counter = 0

            if self.fixed:
                self.trace_idx += 1
                if self.trace_idx >= len(self.all_cooked_time):
                    self.trace_idx = 0
            else:
                # pick a random trace file
                self.trace_idx = np.random.randint(len(self.all_cooked_time))
            self.cooked_time = self.all_cooked_time[self.trace_idx]
            self.cooked_bw = self.all_cooked_bw[self.trace_idx]

            # randomize the start point of the video
            # note: trace file starts with time 0
            self.mahimahi_ptr = self.mahimahi_start_ptr if self.fixed else np.random.randint(1, len(self.cooked_bw))
            self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

        next_video_chunk_sizes = []
        for i in range(BITRATE_LEVELS):
            next_video_chunk_sizes.append(self.video_size[i][self.video_chunk_counter])


        reward = VIDEO_BIT_RATE[bitrate] / M_IN_K \
                 - REBUF_PENALTY * rebuf \
                 - SMOOTH_PENALTY * np.abs( VIDEO_BIT_RATE[bitrate] -
                                            VIDEO_BIT_RATE[self.last_bitrate] ) / M_IN_K

        self.last_bitrate = bitrate
        # dequeue history record
        self.state = np.roll( self.state ,-1 ,axis=1 )

        # this should be S_INFO number of terms
        self.state[0 ,0 ,-1] = VIDEO_BIT_RATE[bitrate] / np.max( VIDEO_BIT_RATE )
        self.state[0 ,1 ,-1] = return_buffer_size / MILLISECONDS_IN_SECOND / \
                               BUFFER_NORM_FACTOR
        self.state[0 ,2 ,-1] = video_chunk_size / delay / M_IN_K  # kbyte/ms
        self.state[0 ,3 ,-1] = delay / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
        self.state[0 ,4 ,:6] = np.array(
            next_video_chunk_sizes ) / M_IN_K / M_IN_K
        self.state[0 ,5 ,-1] = video_chunk_remain / self.total_video_chunk

        debug_info = {'delay': delay ,
                      'sleep_time': sleep_time ,
                      'buffer_size': return_buffer_size / MILLISECONDS_IN_SECOND ,
                      'rebuf': rebuf ,
                      'video_chunk_size': video_chunk_size ,
                      'next_video_chunk_sizes': next_video_chunk_sizes ,
                      'video_chunk_remain': video_chunk_remain}

        return self.state, reward, end_of_video, debug_info
