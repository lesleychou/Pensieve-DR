import os
import random
import subprocess
import sys

from sympy import N, Symbol, solve

# 68 files with 2000 seconds, 205 files with 320 seconds

TRAIN_TRACE_DIR = "../data/generated_traces_lesley/train/train_BW_80-100"
VAL_TRACE_DIR = "../data/generated_traces_lesley/val/val_BW_80-100"
os.makedirs(TRAIN_TRACE_DIR, exist_ok=True)
os.makedirs(VAL_TRACE_DIR, exist_ok=True)

# TEST_TRACE_DIR = "../data/synthetic_test"
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
MIN_THROUGHPUT = 0.2
MAX_THROUGHPUT_LOW = 80
MAX_THROUGHPUT_HIGH = 100
STEPS = 15

cmds = []
processes = []

for i in range(0, 100):
    name = os.path.join(TRAIN_TRACE_DIR, f"trace{i}.txt")
    print("create ", name)
    T_s = T_s
    T_l = T_l
    cov = cov
    duration = duration
    max_throughput = round( random.uniform( MAX_THROUGHPUT_LOW ,MAX_THROUGHPUT_HIGH ) )
    # for T_s experiment:
    # max_throughput = MAX_THROUGHPUT
    min_throughput = MIN_THROUGHPUT
    cmd = "python synthetic_lesley.py --T_l {} --T_s {} --cov {} " \
          "--duration {} --max-throughput {} " \
          "--min-throughput {} --output_file {}".format(
        T_l ,T_s ,cov ,duration ,max_throughput ,min_throughput ,name )
    cmds.append( cmd.split( " " ) )

for i in range(100, 300):
    name = os.path.join(VAL_TRACE_DIR, f"trace{i}.txt")
    print("create ", name)
    T_s = T_s
    T_l = T_l
    cov = cov
    duration = duration
    max_throughput = round(random.uniform(MAX_THROUGHPUT_LOW, MAX_THROUGHPUT_HIGH))
    # for T_s experiment:
    #max_throughput = MAX_THROUGHPUT
    min_throughput = MIN_THROUGHPUT
    cmd = "python synthetic_lesley.py --T_l {} --T_s {} --cov {} " \
        "--duration {} --max-throughput {} " \
        "--min-throughput {} --output_file {}".format(
                T_l, T_s, cov, duration, max_throughput, min_throughput, name)
    cmds.append(cmd.split(" "))

# for x in range(1, 100):
#     MAX_THROUGHPUT_HIGH = x
#     for i in range(0, 50):
#         os.makedirs(TEST_TRACE_DIR+"/"+str(x), exist_ok=True)
#         name = os.path.join(TEST_TRACE_DIR+"/"+str(x), f"trace{i}.txt")
#         print("create ", name)
#         T_s = T_s
#         T_l = T_l
#         cov = cov
#         duration = duration
#         #max_throughput = round(random.uniform(MIN_THROUGHPUT, MAX_THROUGHPUT),1)
#         #for T_s experiment:
#         min_throughput = MIN_THROUGHPUT
#         max_throughput = MAX_THROUGHPUT_HIGH
#         cmd = "python synthetic_lesley.py --T_l {} --T_s {} --cov {} " \
#             "--duration {} --steps {} --switch-parameter {} --max-throughput {} " \
#             "--min-throughput {} --output_file {}".format(
#                     T_l, T_s, cov, duration, STEPS, switch_parameter,
#                     max_throughput, min_throughput, name)
#         cmds.append(cmd.split(' '))

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