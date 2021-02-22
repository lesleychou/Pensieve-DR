import os
import random
import subprocess
import sys

from sympy import N, Symbol, solve

# 68 files with 2000 seconds, 205 files with 320 seconds

TRAIN_TRACE_DIR = "../data/generated_traces_ts_float-BO/train-2-iter/train_3-5"
os.makedirs(TRAIN_TRACE_DIR, exist_ok=True)

#VAL_TRACE_DIR = "../data/generated_traces_huge/val_0-500/val_BW_450-550"
#os.makedirs(VAL_TRACE_DIR, exist_ok=True)

# TEST_TRACE_DIR = "../data/synthetic_test_lesley_3"
# os.makedirs(TEST_TRACE_DIR, exist_ok=True)

# T_s_min = 10
# T_s_max = 100
# T_l_min = 1
# T_l_max = 5
# cov_min = 0.05
# cov_max = 0.5
# duration_min = 320
# duration_max = 2000

# 1-200000
# 200000-400000
# 400000-600000
# 600000-800000
# 800000-1000000

# 1-500
# 500 -1000
# 1000 - 240000
# 240000-640000
# 640000-1000000

# large range
T_s = 3
T_l = 5
cov = 0.01
duration = 300
MAX_TASK_CNT = 32
MIN_THROUGHPUT = 2
MAX_THROUGHPUT_LOW = 3
MAX_THROUGHPUT_HIGH = 5
STEPS = 15

cmds = []
processes = []

for i in range(0, 200):
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

# for i in range(100, 200):
#     name = os.path.join(VAL_TRACE_DIR, f"trace{i}.txt")
#     print("create ", name)
#     T_s = T_s
#     T_l = T_l
#     cov = cov
#     duration = duration
#     max_throughput = round(random.uniform(MAX_THROUGHPUT_LOW, MAX_THROUGHPUT_HIGH))
#     # for T_s experiment:
#     #max_throughput = MAX_THROUGHPUT
#     min_throughput = MIN_THROUGHPUT
#     cmd = "python synthetic_lesley.py --T_l {} --T_s {} --cov {} " \
#         "--duration {} --max-throughput {} " \
#         "--min-throughput {} --output_file {}".format(
#                 T_l, T_s, cov, duration, max_throughput, min_throughput, name)
#     cmds.append(cmd.split(" "))

# for x in range(1, 100):
#     MAX_THROUGHPUT_HIGH = x
#     MAX_THROUGHPUT_LOW = x
#     for i in range(0, 50):
#         os.makedirs( TEST_TRACE_DIR + "/" + str( x ) ,exist_ok=True )
#         name = os.path.join(TEST_TRACE_DIR+"/"+str(x), f"trace{i}.txt")
#         print("create ", name)
#         T_s = T_s
#         T_l = T_l
#         cov = cov
#         duration = duration
#         max_throughput = round(random.uniform(MAX_THROUGHPUT_LOW, MAX_THROUGHPUT_HIGH))
#         min_throughput = MIN_THROUGHPUT
#         cmd = "python synthetic_lesley.py --T_l {} --T_s {} --cov {} " \
#             "--duration {} --max-throughput {} " \
#             "--min-throughput {} --output_file {}".format(
#                     T_l, T_s, cov, duration, max_throughput, min_throughput, name)
#         cmds.append(cmd.split(" "))

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