import os
import random
import subprocess
import sys

from sympy import N, Symbol, solve

# 68 files with 2000 seconds, 205 files with 320 seconds

TRAIN_TRACE_DIR = "../data/tmp/train/train_generate_2_BW_0-80"
#VAL_TRACE_DIR = "../data/generated_traces_2/val/val_BW_0-90"
# TEST_TRACE_DIR = "../data/synthetic_test/jump-action-test-BW-1000"

os.makedirs(TRAIN_TRACE_DIR, exist_ok=True)
#os.makedirs(VAL_TRACE_DIR, exist_ok=True)
# os.makedirs(TEST_TRACE_DIR, exist_ok=True)

# T_s_min = 10
# T_s_max = 100
# T_l_min = 1
# T_l_max = 5
# cov_min = 0.05
# cov_max = 0.5
# duration_min = 320
# duration_max = 2000


# large range
T_s = 5
T_l = 5
cov = 0.01
duration = 250
MAX_TASK_CNT = 32
MIN_THROUGHPUT = 80
MAX_THROUGHPUT = 100
STEPS = 6

cmds = []
processes = []
eq = -1
x = Symbol("x", positive=True)
for y in range(1, STEPS-1):
    eq += (1/x**y)
res = solve(eq, x)
switch_parameter = N(res[0])
for i in range(0, 60):
    name = os.path.join(TRAIN_TRACE_DIR, f"trace_100_{i}.txt")
    print("create ", name)
    T_s = T_s
    T_l = T_l
    cov = cov
    duration = duration
    #max_throughput = round(random.uniform(MIN_THROUGHPUT, MAX_THROUGHPUT),1)
    # for T_s experiment:
    max_throughput = MAX_THROUGHPUT
    min_throughput = MIN_THROUGHPUT
    cmd = "python synthetic_traces.py --T_l {} --T_s {} --cov {} " \
        "--duration {} --steps {} --switch-parameter {} --max-throughput {} " \
        "--min-throughput {} --output_file {}".format(
                T_l, T_s, cov, duration, STEPS, switch_parameter,
                max_throughput, min_throughput, name)
    cmds.append(cmd.split(" "))

# for i in range(30, 40):
#     name = os.path.join(VAL_TRACE_DIR, f"trace{i}.txt")
#     print("create ", name)
#     T_s = T_s
#     T_l = T_l
#     cov = cov
#     duration = duration
#     # max_throughput = round(random.uniform(MIN_THROUGHPUT, MAX_THROUGHPUT),1)
#     # for T_s experiment:
#     max_throughput = MAX_THROUGHPUT
#     min_throughput = MIN_THROUGHPUT
#     cmd = "python synthetic_traces.py --T_l {} --T_s {} --cov {} " \
#         "--duration {} --steps {} --switch-parameter {} --max-throughput {} " \
#         "--min-throughput {} --output_file {}".format(
#                 T_l, T_s, cov, duration, STEPS, switch_parameter,
#                 max_throughput, min_throughput, name)
#     cmds.append(cmd.split(" "))

while True:
    while cmds and len(processes) < MAX_TASK_CNT:
        cmd = cmds.pop()
        processes.append(subprocess.Popen(cmd, stdin=open(os.devnull)))
    for p in processes:
        if p.poll() is not None:
            if p.returncode == 0:
                print(p.args, 'finished!')
                processes.remove(p)
            else:
                print(p.args, 'failed!')
                sys.exit(1)

    if not processes and not cmds:
        break