import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})

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







