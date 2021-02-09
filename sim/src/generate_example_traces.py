"""
This is just an example of how to generate some random traces....
"""
import subprocess

class TraceConfig:
    def __init__(self,
                 trace_dir,                 
                 max_throughput=10):
        self.trace_dir = trace_dir
        self.max_throughput = max_throughput
        self.T_l = 0
        self.T_s = 3
        self.cov = 3
        self.duration = 250
        self.step = 0
        self.min_throughput = 0.2
        self.num_traces = 100

def generate_traces_with(config):
    """
    Generates traces based on the config
    """
    script = "./trace_generator.py"
    command = "python {script} \"{config}\"".format(script=script, config=vars(config))
    # alternatively call with os.system, but it doesn't print the result that way
    # os.system(command)    
    output = subprocess.check_output(command, shell=True, text=True).strip()
    print(output)

config = TraceConfig("example_traces/")
generate_traces_with(config)
