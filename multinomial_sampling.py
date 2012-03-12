from numpy.random import random_sample


def sample(weights):
    """Sample a number from 0 to len(weights)-1 where the probability
    of sampling i is proportional to weights[i]."""
    u = random_sample() * sum(weights)
    sample = 0
    weight_sum = weights[0]
    while sample < len(weights) and weight_sum <= u:
        sample += 1
        weight_sum += weights[sample]
    return sample
