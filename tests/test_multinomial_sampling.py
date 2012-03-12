from .. import multinomial_sampling

def test_sample():
    # Check that numbers are sampled in the right proportions
    weights = [1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]
    num_samples = 160000
    epsilon = 0.01
    answer = 6.0
    avg = 0.0

    for i in xrange(num_samples):
        avg += multinomial_sampling.sample(weights) / float(num_samples)
    print(avg)
    assert abs(avg - answer) < epsilon
