'''
MORL helper functions
'''
from src.reward import reward_scalar_calculation
import numpy as np

def envelope_operator(args, preference, target, value, global_step):
    '''
    Runs the envelope operator calculation outlined in

    "A Generalized Algorithm for Multi-Objective Reinforcement Learning and Policy Adaptation"

    See

    https://github.com/RunzheYang/MORL

    for more info.
    '''

    # Data has the following shape.
    # [w1, w1, w1, w1, w1, w1,    w2, w2, w2, w2, w2, w2...]
    # [s1, s2, s3, u1, u2, u3,    s1, s2, s3, u1, u2, u3...]
    
    agents_times_gae = args.NUM_AGENTS * args.NUM_GAE_STEP
    target = np.concatenate(target).reshape(-1, args.REWARD_SIZE)

    # If our step count is past where we have set it to start, then
    # perform more edits on the target.
    if global_step > args.ENVELOPE_START:
        # Improvement: Log that we are running an envelope calculation
        product = np.inner(target, preference)
        envelope_mask = product.transpose().reshape(args.SAMPLE_SIZE, -1, agents_times_gae).argmax(axis=1)
        envelope_mask = envelope_mask.reshape(-1) * agents_times_gae + np.array(list(range(agents_times_gae)) * args.SAMPLE_SIZE)
        target = target[envelope_mask]

    # For the actor,
    #   Q = state Value function V(s) + advantage value A(s, a)
    #   adv = Q - state Value function V(s)
    #   value: Critic value given states
    adv = target - value

    return target, adv

def calculate_discounted_return(reward, terminal, value, next_value, reward_size, gae_steps, use_gae=False, gamma=0.99, lam=0.95):
    '''
    Calculates discounted return from parameters.
    '''

    # Initialize a discounted_return numpy array of size
    # number of steps, reward_size
    discounted_return = np.empty([gae_steps, reward_size])

    # Discounted Return
    if use_gae:
        # Generalized Advantage Estimator
        gae = np.zeros(reward_size)
        for time_step in range(gae_steps - 1, -1, -1):
            # delta is the difference between what the model predicts
            # and what the target model is predicting
            delta = reward[time_step] + gamma * next_value[time_step] * (1 - terminal[time_step]) - value[time_step]
            gae = delta + gamma * lam * (1 - terminal[time_step]) * gae
            discounted_return[time_step] = gae + value[time_step]

    else:
        # Start with the last next value.
        running_add = next_value[-1]

        # Iterate backwards in time over the number of steps.
        for time_step in range(gae_steps - 1, -1, -1):
            # The running addition at each time step becomes the
            # reward at that time step + a discounted figure of the
            # running_add.
            running_add = reward[time_step] + gamma * running_add * (1 - terminal[time_step])
            discounted_return[time_step] = running_add

    return discounted_return
