import numpy as np
import torch
import copy
from scipy.spatial.distance import pdist, squareform
from WFA import build_WFA, run_WFA, calculate_average_input_distance

def get_distant_samples(X, n_samples):
    """
    Select the n_samples most distant samples from X.

    Arguments:
        X: array of shape (n_samples, n_features)
        n_samples: number of samples to select
    """

    # Compute the pairwise distances between samples
    distances = squareform(pdist(X))

    # Compute the sum of distances for each sample
    sum_distances = distances.sum(axis=1)

    # Select the n_samples samples with the largest sum of distances
    indices = np.argpartition(sum_distances, -n_samples)[-n_samples:]

    return X[indices]

def generate_adversarial_sample_SA(model, sample, target_sample, minimum_positive_sample, label, eps, T_initial, T_final, cooling_rate, sample_weights):
    """
    Generate adversarial sample using Simulated Annealing.

    Arguments:
        model: target rnn classifier
        sample: a single time series sample to be perturbed
        target_sample: the target positive sample
        minimum_positive_sample: the minimum positive sample
        label: the true label of the sample
        eps: maximum allowed perturbation
        T_initial: initial temperature for SA
        T_final: final temperature for SA
        cooling_rate: cooling rate for SA, should be in (0, 1)
        sample_weights: the weights for perturbation of the sample
    """

    # Define the energy function
    def energy(x):
        prediction = model(torch.from_numpy(x).to(torch.float32).unsqueeze(0)).argmax().item()
        prediction_error = 0 if prediction == label else 1
        target_distance = np.linalg.norm(x - target_sample)
        minimum_distance = np.linalg.norm(x - minimum_positive_sample)
        return prediction_error + target_distance - minimum_distance

    # Check if the weights are all zero
    if np.all(sample_weights == 0):
        # Compute the derivative of the sample
        sample_derivative = np.gradient(sample)

        # Normalize the derivative to [0, 1] to get the weights
        weights = (sample_derivative - sample_derivative.min()) / (sample_derivative.max() - sample_derivative.min())
    else:
        weights = sample_weights

    # Initialize SA parameters
    T = T_initial
    current_sample = copy.deepcopy(sample)
    current_energy = energy(current_sample)

    # SA loop
    while T > T_final:
        # Generate a candidate sample with weighted perturbation
        candidate_sample = current_sample + weights * np.random.uniform(-eps, eps, size=current_sample.shape)
        candidate_energy = energy(candidate_sample)

        # Acceptance probability
        p = np.exp((current_energy - candidate_energy) / T)
        if p > np.random.rand():
            current_sample = candidate_sample
            current_energy = candidate_energy

        # Cooling
        T *= cooling_rate

    return current_sample

def TSFool_SA(model, X, Y, weights, K=2, T=30, F=0.1, eps=0.01, N=20, P=0.9, C=1, target=-1, details=False,
              T_initial=10, T_final=0.1, cooling_rate=0.99):
    """
    Implementation of the TSFool attack using Simulated Annealing for adversarial sample generation.

    The arguments are the same as the original TSFool function, with three additional arguments for SA:
        T_initial: initial temperature for SA
        T_final: final temperature for SA
        cooling_rate: cooling rate for SA, should be in (0, 1)
    """

    # Initialize the output arrays
    adv_X = np.empty(X.shape)
    adv_Y = np.empty(Y.shape)
    target_positive_sample_X = np.empty((N, X.shape[1]))
    adv_index = np.empty(N)

    # Loop over the samples
    for i in range(len(X)):
        # Select the target positive sample and minimum positive sample
        target_positive_sample = get_target_positive_sample(model, X[i], Y[i])
        minimum_positive_sample = get_minimum_positive_sample(model, X[i], Y[i])

        # Generate the adversarial sample using Simulated Annealing
        adv_x = generate_adversarial_sample_SA(model, X[i], target_positive_sample, minimum_positive_sample, Y[i], eps, T_initial, T_final, cooling_rate, weights[i])

        # Update the output arrays
        adv_X[i] = adv_x
        adv_Y[i] = model.predict(adv_x)
        target_positive_sample_X[i] = target_positive_sample
        adv_index[i] = i

    return adv_X, adv_Y, target_positive_sample_X, adv_index

if __name__ == '__main__':
    # Load your model and dataset here
    from models.models_structure.ECG200 import RNN
    model = torch.load('models/ECG200.pkl')
    dataset_name = 'ECG200'
    X = np.load(f'datasets/preprocessed/{dataset_name}/{dataset_name}_TEST_X.npy')
    Y = np.load(f'datasets/preprocessed/{dataset_name}/{dataset_name}_TEST_Y.npy')
    weights = np.loadtxt('weights.csv', delimiter=',')

    # Select the most distant samples
    X = get_distant_samples(X, K)

    # Call the modified TSFool function
    adv_X, adv_Y, target_X, index = TSFool_SA(model, X, Y, weights, K=2, T=30, F=0.1, eps=0.01, N=20, P=0.9, C=1, target=-1, details=False,
                                              T_initial=10, T_final=0.1, cooling_rate=0.99)

    print(index)
