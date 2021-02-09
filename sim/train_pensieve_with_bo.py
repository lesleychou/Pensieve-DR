# This is the main training setup for training a Pensieve model, but
# stopping at intervals to add new training data based on performance
# (increases generalizability, we hope!)

# Inputs:
#
# - experiment results directory
# - training data directory (with traces in subdirectories)
# - total epochs
# - bayesian optimizer interval (e.g., every 5000 epochs) - this is how many epoch each training run will go

# Defaults
# Improvement: Probably better if replaced with argparse and passed in (later)
TOTAL_EPOCHS = 10000
BAYESIAN_OPTIMIZER_INTERVAL = 1000
TRAINING_DATA_DIRECTORY = "../data/training_default/"
RESULTS_DIR = "../results/bo_example/"

num_training_runs = int(TOTAL_EPOCHS / BAYESIAN_OPTIMIZER_INTERVAL)

# Example Flow:
for i in range(num_training_runs):
    if i > 0:
        print("Running bayesian optimization to get new training data input parameter.")
        print("Run test generation based on BO output and put in training data dir.")

    print("Check results dir for any saved model file.")
    print("If it exists, take the latest.")
    print("Get the file and pass it to the training script, if it exists.\n")
    print("Running training:", i)

print("Hooray!")
