import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})

#RESULTS_FOLDER = './results/norway-PPO/'
RESULTS_FOLDER = '../easy-param-results/default_first_2/seed_1/'
NUM_BINS = 100
BITS_IN_BYTE = 8.0
MILLISEC_IN_SEC = 1000.0
M_IN_B = 1000000.0
VIDEO_LEN = 48
VIDEO_BIT_RATE = [300, 1200, 2850, 6500, 33000, 165000]  # Kbps
K_IN_M = 1000.0
REBUF_P = 165
SMOOTH_P = 1
COLOR_MAP = plt.cm.jet #nipy_spectral, Set1,Paired 
SIM_DP = 'sim_dp'
#SCHEMES = ['BB', 'RB', 'FIXED', 'FESTIVE', 'BOLA', 'RL',  'sim_rl', SIM_DP]
#SCHEMES = ['sim_bb', 'sim_mpc', 'sim_rl_pretrain', 'sim_rl_train_noise001', 'sim_rl_train_noise002', 'sim_rl_train_noise003']
SCHEMES = ['sim_mpc', 'sim_adr']
#SCHEMES = ['sim_rl']



def compute_cdf(data):
    """ Return the cdf of input data.

    Args
        data(list): a list of numbers.

    Return
        sorted_data(list): sorted list of numbers.

    """
    length = len(data)
    sorted_data = sorted(data)
    cdf = [i / length for i, val in enumerate(sorted_data)]
    return sorted_data, cdf


def main():
    time_all = {}
    bit_rate_all = {}
    buff_all = {}
    bw_all = {}
    raw_reward_all = {}

    for scheme in SCHEMES:
        time_all[scheme] = {}
        raw_reward_all[scheme] = {}
        bit_rate_all[scheme] = {}
        buff_all[scheme] = {}
        bw_all[scheme] = {}

    log_files = os.listdir(RESULTS_FOLDER)
    for log_file in log_files:

        time_ms = []
        bit_rate = []
        buff = []
        bw = []
        reward = []

        #print(log_file)

        with open(RESULTS_FOLDER + log_file, 'r') as f:
            if SIM_DP in log_file:
                last_t = 0
                last_b = 0
                last_q = 1
                lines = []
                for line in f:
                    lines.append(line)
                    parse = line.split()
                    if len(parse) >= 6:
                        time_ms.append(float(parse[3]))
                        bit_rate.append(VIDEO_BIT_RATE[int(parse[6])])
                        buff.append(float(parse[4]))
                        bw.append(float(parse[5]))

                for line in reversed(lines):
                    parse = line.split()
                    r = 0
                    if len(parse) > 1:
                        t = float(parse[3])
                        b = float(parse[4])
                        q = int(parse[6])
                        if b == 4:
                            rebuff = (t - last_t) - last_b
                            assert rebuff >= -1e-4
                            r -= REBUF_P * rebuff

                        r += VIDEO_BIT_RATE[q] / K_IN_M
                        r -= SMOOTH_P * np.abs(VIDEO_BIT_RATE[q] - VIDEO_BIT_RATE[last_q]) / K_IN_M
                        reward.append(r)

                        last_t = t
                        last_b = b
                        last_q = q

            else:
                for line in f:
                    parse = line.split()
                    if len(parse) <= 1:
                        break
                    time_ms.append(float(parse[0]))
                    bit_rate.append(int(parse[1]))
                    buff.append(float(parse[2]))
                    bw.append(float(parse[4]) / float(parse[5]) * BITS_IN_BYTE * MILLISEC_IN_SEC / M_IN_B)
                    reward.append(float(parse[6]))
                #print( reward, "--------------------" )


        if SIM_DP in log_file:
            time_ms = time_ms[::-1]
            bit_rate = bit_rate[::-1]
            buff = buff[::-1]
            bw = bw[::-1]

        time_ms = np.array(time_ms)
        time_ms -= time_ms[0]

        # print log_file

        for scheme in SCHEMES:
            if scheme in log_file:
                time_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = time_ms
                bit_rate_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = bit_rate
                buff_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = buff
                bw_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = bw
                raw_reward_all[scheme][log_file[len('log_' + str(scheme) + '_'):]] = reward
                break

    # ---- ---- ---- ----
    # Reward records
    # ---- ---- ---- ----

    log_file_all = []
    reward_all = {}
    for scheme in SCHEMES:
        reward_all[scheme] = []


    for l in time_all[SCHEMES[0]]:
        # what is l here?
        # l will be something like "norway_ferry_7", representing the name of a trace
        # print(l)

        # assume that the schemes are okay, then flip the flag if they are not
        schemes_check = True

        # all schemes must pass the check
        for scheme in SCHEMES:
            # print(l not in time_all[scheme])
            # check 1: l is a trace name. is the trace name found in every scheme? if not, we fail
            # check 2: is the length of the log for trace "l" less than the video length? if not, we fail
            if l not in time_all[scheme] or len(time_all[scheme][l]) < VIDEO_LEN:
                # print all the bad ls
                # print(l)
                # print(scheme)
                schemes_check = False
                break
        if schemes_check:
            log_file_all.append(l)
            for scheme in SCHEMES:
                #print(raw_reward_all[scheme], "----------------------")
                reward_all[scheme].append(np.sum(raw_reward_all[scheme][l][1:VIDEO_LEN])/VIDEO_LEN)
    #print(reward_all[scheme], scheme)


    mean_rewards = {}
    error_bar = {}
    for scheme in SCHEMES:
        mean_rewards[scheme] = np.mean(reward_all[scheme])
        mean_rewards[scheme] = round(mean_rewards[scheme], 3)
        error_bar[scheme] = np.var(reward_all[scheme])
        error_bar[scheme] = round(error_bar[scheme], 2)


    fig = plt.figure()
    ax = fig.add_subplot(111)


    for scheme in SCHEMES:
        ax.plot(reward_all[scheme])

    SCHEMES_REW = []
    for scheme in SCHEMES:
        SCHEMES_REW.append(scheme + ': ' + str(mean_rewards[scheme])  + '% ' + str(error_bar[scheme]))

    colors = [COLOR_MAP(i) for i in np.linspace(0, 1, len(ax.lines))]
    for i,j in enumerate(ax.lines):
        j.set_color(colors[i])

    ax.legend(SCHEMES_REW)
    print(SCHEMES_REW)
    plt.ylabel('Mean reward')
    plt.xlabel('trace index')
    plt.title('Real-trace: Norway')
    plt.show()

    # ---- ---- ---- ----
    # CDF
    # ---- ---- ---- ----

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for scheme in SCHEMES:
        sorted_data, cdf = compute_cdf(reward_all[scheme])
        ax.plot(sorted_data, cdf )

    colors = [COLOR_MAP(i) for i in np.linspace(0, 1, len(ax.lines))]
    for i,j in enumerate(ax.lines):
        j.set_color(colors[i])

    ax.legend(SCHEMES_REW)

    #plt.xlim(-0.5, 2.5)

    plt.ylabel('CDF')
    plt.xlabel('total reward')
    plt.title('CDF on real-trace: Norway')
    plt.show()

    # plot the Pensieve-MPC
    # difference = []
    # zip_object = zip( reward_all['sim_rl'], reward_all['sim_bb'] )
    # for list1_i, list2_i in zip_object:
    # 	difference.append( list1_i - list2_i )
    # print( difference )
    #
    # hist, bin_edges = np.histogram( difference, bins=10 )
    # cdf = np.cumsum( hist )
    # ax.legend( "Pensieve-MPC" )
    # plt.xlim( -30, 60 )
    # plt.plot( bin_edges[1:], cdf / cdf[-1] )
    # plt.show()


    # ---- ---- ---- ----
    # check each trace
    # ---- ---- ---- ----
    count = 0
    for l in time_all[SCHEMES[0]]:
        schemes_check = True
        for scheme in SCHEMES:
            if l not in time_all[scheme] or len(time_all[scheme][l]) < VIDEO_LEN:
                schemes_check = False
                break
        if schemes_check:
            fig = plt.figure()

            ax = fig.add_subplot(311)
            for scheme in SCHEMES:
                ax.plot(time_all[scheme][l][:VIDEO_LEN], bit_rate_all[scheme][l][:VIDEO_LEN])
            colors = [COLOR_MAP(i) for i in np.linspace(0, 1, len(ax.lines))]
            for i,j in enumerate(ax.lines):
                j.set_color(colors[i])	
            plt.title(l)
            plt.ylabel('bit rate selection (kbps)')

            ax = fig.add_subplot(312)
            for scheme in SCHEMES:
                ax.plot(time_all[scheme][l][:VIDEO_LEN], buff_all[scheme][l][:VIDEO_LEN])
            colors = [COLOR_MAP(i) for i in np.linspace(0, 1, len(ax.lines))]
            for i,j in enumerate(ax.lines):
                j.set_color(colors[i])	
            plt.ylabel('buffer size (sec)')

            ax = fig.add_subplot(313)
            for scheme in SCHEMES:
                ax.plot(time_all[scheme][l][:VIDEO_LEN], bw_all[scheme][l][:VIDEO_LEN])
            colors = [COLOR_MAP(i) for i in np.linspace(0, 1, len(ax.lines))]
            for i,j in enumerate(ax.lines):
                j.set_color(colors[i])	
            plt.ylabel('bandwidth (mbps)')
            plt.xlabel('time (sec)')

            SCHEMES_REW = []
            for scheme in SCHEMES:
                SCHEMES_REW.append(scheme + ': ' + str(np.sum(raw_reward_all[scheme][l][1:VIDEO_LEN])))

            # rl_reward = np.sum(raw_reward_all["sim_Pensieve"][l][1:VIDEO_LEN])
            adr_reward = np.sum(raw_reward_all["sim_adr"][l][1:VIDEO_LEN])
            mpc_reward = np.sum(raw_reward_all["sim_mpc"][l][1:VIDEO_LEN])

            # if rl_reward - mpc_reward < -50:
            #     count+=1
            #     print(l)

            ax.legend(SCHEMES_REW, loc=3, bbox_to_anchor=(0.5, -0.1), ncol=int(np.ceil(len(SCHEMES) / 3.0)))
            plt.show()



if __name__ == '__main__':
    main()
