import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})

"""
#####################
VIDEO_CHUNCK_LEN = 2000.0
# ['sim_mpc: -21.7% 1120.22', 'sim_adr: -13.17% 805.52']

VIDEO_CHUNCK_LEN = 4000.0
# ['sim_mpc: -4.26% 74.4', 'sim_adr: 0.66% 0.04']

VIDEO_CHUNCK_LEN = 6000.0
# ['sim_mpc: -2.15% 58.89', 'sim_adr: 0.88% 0.03']

VIDEO_CHUNCK_LEN = 8000.0
# ['sim_mpc: -0.7% 118.59', 'sim_adr: 1.02% 0.03']

#####################
VIDEO_BIT_RATE = [300, 1200, 2850, 6500, 33000, 165000]
# ['sim_mpc: -4.26% 74.4', 'sim_adr: 0.66% 0.04']

VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]
# ['sim_mpc: 0.62% 0.11', 'sim_adr: 0.68% 0.04']

VIDEO_BIT_RATE = [300, 950, 1850, 3900, 7800, 15600]
# ['sim_mpc: 0.477% 0.42', 'sim_adr: 0.696% 0.15']

VIDEO_BIT_RATE = [300, 750, 1850, 4300, 15600, 33000]
# ['sim_mpc: 0.201% 1.82', 'sim_adr: 0.696% 0.05']


#####################
VIDEO_START_PLAY = 2000.0, 4000.0
# ['sim_mpc: -3.556% 86.56', 'sim_adr: 0.665% 0.04']

VIDEO_START_PLAY = 8000.0
# ['sim_mpc: -3.519% 96.86', 'sim_adr: 0.674% 0.04']

VIDEO_START_PLAY = 16000.0
# ['sim_mpc: -1.933% 73.48', 'sim_adr: 0.683% 0.04']
"""
fig = plt.figure()
ax = fig.add_subplot(111)

width = 0.1
x = [ 1, 2, 3, 4 ]

x1 = [p - 0.2 for p in x]
x2 = [p for p in x]

# VIDEO_START_PLAY
adr = [0.665, 0.665, 0.674, 0.683]
adr_err = [0.04, 0.04, 0.04, 0.04]
mpc = [-3.56, -3.56, -3.51, -1.93]
mpc_err = [0.8, 0.7, 0.9, 0.7]
labels = [ '2s', '4s', '8s', '16s']

# bitrate level
# adr = [0.66, 0.66, 0.696, 0.696]
# adr_err = [0.04, 0.04, 0.15, 0.05]
# mpc = [-4.26, 0.62, 0.477, 0.201]
# mpc_err = [1.7, 0.11, 0.22, 0.18]
# labels = [ '1st combo', '2nd combo', '3rd combo', '4th combo']

# video-chunk-length
# adr = [-13.17, 0.66, 0.88, 1.02]
# adr_err = [2.8, 0.04, 0.03, 0.03]
# mpc = [-21.7, -4.26, -2.15, -0.7]
# mpc_err = [3.1, 0.7, 0.5, 0.11]
# labels = [ '2s', '4s', '6s', '8s']


ax.bar(x1, adr, yerr=adr_err, width=0.2,color='red', alpha=0.6, label="adr")
ax.bar(x2, mpc, yerr=mpc_err, width=0.2,color='green', alpha=0.8, label="mpc")


#labels = [[0,2], [0,5], [0,20], [0,60]]

ax.set_xticks( x )
ax.set_xticklabels( labels )

ax.legend(loc=4)

#ax.set_xlim(5,15)
plt.ylabel('Mean reward per chunk')
plt.xlabel('Initial waiting time (seconds)')
plt.title('Generalizability on initial-waiting-time(when will the video play)')
plt.show()







