import numpy as np

MILLISECONDS_IN_SECOND = 1000.0
B_IN_MB = 1000000.0
BITS_IN_BYTE = 8.0
RANDOM_SEED = 42
VIDEO_CHUNCK_LEN = 4000.0  # millisec, every time add this amount to buffer
BITRATE_LEVELS = 6
TOTAL_VIDEO_CHUNCK = 48
PACKET_SIZE = 1500  # bytes
NOISE_LOW = 0.9
NOISE_HIGH = 1.1
VIDEO_SIZE_FILE = '../data/video_size_6_larger/video_size_'

# # Env params need to do UDR
# BUFFER_THRESH = 60000.0     # 60.0 * MILLISECONDS_IN_SECOND, max buffer limit
# DRAIN_BUFFER_SLEEP_TIME = 500.0    # millisec
# PACKET_PAYLOAD_PORTION = 0.95
# LINK_RTT = 80  # millisec


class Environment:
    def __init__(self, buffer_thresh, drain_buffer_sleep_time,
                 packet_payload_portion, link_rtt,
                 all_cooked_time, all_cooked_bw, all_file_names=None,
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

        # all UDR params
        self.buffer_thresh = buffer_thresh
        self.drain_buffer_sleep_time = drain_buffer_sleep_time
        self.packet_payload_portion = packet_payload_portion
        self.link_rtt = link_rtt
        print(self.link_rtt, "------link rtt")

    def get_video_chunk(self, quality):

        assert quality >= 0
        assert quality < BITRATE_LEVELS

        video_chunk_size = self.video_size[quality][self.video_chunk_counter]
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

            packet_payload = throughput * duration * self.packet_payload_portion

            if video_chunk_counter_sent + packet_payload > video_chunk_size:
                fractional_time = (video_chunk_size - video_chunk_counter_sent) / \
                                  throughput / self.packet_payload_portion
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

        delay += self.link_rtt
        #print( delay ,"--------RTT------" )

        # add a multiplicative noise to the delay
        if not self.fixed:
            delay *= np.random.uniform(NOISE_LOW, NOISE_HIGH)

        # rebuffer time
        rebuf = np.maximum(delay - self.buffer_size, 0.0)

        # update the buffer
        # the initial
        # if self.buffer_size>8 && 1st video chunk, else pass
        self.buffer_size = np.maximum(self.buffer_size - delay, 0.0)

        # add in the new chunk
        self.buffer_size += VIDEO_CHUNCK_LEN

        # sleep if buffer gets too large
        sleep_time = 0
        if self.buffer_size > self.buffer_thresh:
            # exceed the buffer limit
            # we need to skip some network bandwidth here
            # but do not add up the delay
            drain_buffer_time = self.buffer_size - self.buffer_thresh
            sleep_time = np.ceil(drain_buffer_time / self.drain_buffer_sleep_time) * \
                         self.drain_buffer_sleep_time
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
        video_chunk_remain = TOTAL_VIDEO_CHUNCK - self.video_chunk_counter

        end_of_video = False
        if self.video_chunk_counter >= TOTAL_VIDEO_CHUNCK:
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

        return delay, \
            sleep_time, \
            return_buffer_size / MILLISECONDS_IN_SECOND, \
            rebuf / MILLISECONDS_IN_SECOND, \
            video_chunk_size, \
            next_video_chunk_sizes, \
            end_of_video, \
            video_chunk_remain