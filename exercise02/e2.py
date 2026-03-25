# Load the data and libraries
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pytest
plt.style.use('seaborn-v0_8-whitegrid')

# question 1
def laplace_mech(v, sensitivity, epsilon):
    """
    This function adds noise drawn from a Laplace distribution to the 
    original query results to provide mathematical privacy guarantees.

    Args:
        v (float or ndarray): The raw query result (true value). 
            Can be a single scalar or a NumPy array of values.
        sensitivity (float): The l1-sensitivity of the query function. 
            Represents the maximum change in the output 'v' caused by 
            adding or removing a single entry in the dataset.
        epsilon (float): The privacy budget. A smaller epsilon provides 
            stronger privacy but adds more noise (less utility). 
            Must be greater than 0.

    Returns:
        float or ndarray: The privatized result (v + noise), 
            maintaining the same shape as the input 'v'.
    """
    noise = np.random.laplace(loc=0, scale=sensitivity / epsilon, size=np.shape(v))
    return v + noise


def test_laplace_mech_scalar_shape():
    """Output shape matches scalar input."""
    result = laplace_mech(5.0, sensitivity=1.0, epsilon=1.0)
    assert np.shape(result) == ()


def test_laplace_mech_vector_shape():
    """Output shape matches vector input."""
    v = np.array([1.0, 2.0, 3.0])
    result = laplace_mech(v, sensitivity=1.0, epsilon=1.0)
    assert result.shape == v.shape



def avg_wages(data, epsilon):
    """
    Calculates the differentially private average of hourly wages using 
    the Laplace Mechanism with global sensitivity clipping.

    The function clips wage values to a fixed range [0, 300] to bound the 
    L1-sensitivity, ensuring that no single individual's wage can 
    disproportionately affect the sum.

    Args:
        data (pd.DataFrame): The input dataset containing a 'HRLYEARN' column 
            representing hourly earnings.
        epsilon (float): The privacy budget (epsilon). Determines the amount 
            of noise added; smaller values provide stronger privacy.

    Returns:
        float: The privatized average hourly wage.

    Notes:
        - Sensitivity: For a sum query with clipped values [0, B], 
          the sensitivity is equal to B (the clipping_parameter).
        - Bias: Clipping may introduce negative bias if many true wages 
          exceed the 300 cap, but it is necessary to bound sensitivity.
    """
    wages = data['HRLYEARN']
    wages = wages.dropna()
    wages = wages / 100         # Unit conversion
    clipping_parameter = 300    # Reasonable Canadian wage cap
    wages = np.clip(wages, a_min=0, a_max=clipping_parameter)
    n = len(wages)

    noisy_sum = laplace_mech(np.sum(wages), sensitivity=clipping_parameter, epsilon=epsilon)
    return noisy_sum / n


def test_avg_wages_output_is_scalar():
    """Result should be a single number."""
    data = pd.DataFrame({'HRLYEARN': [1000, 2000, 3000, 4000]})
    result = avg_wages(data, epsilon=1.0)
    assert np.isscalar(result) or np.shape(result) == ()


def test_avg_wages_ignores_nan():
    """NaN entries should be dropped, not cause NaN output."""
    data = pd.DataFrame({'HRLYEARN': [1000, np.nan, 3000, np.nan]})
    result = avg_wages(data, epsilon=1.0)
    assert not np.isnan(result)


def test_avg_wages_clips_high_values():
    """Values above 30000 (=$300) should be clipped, not dominate the average."""
    data_normal = pd.DataFrame({'HRLYEARN': [2000] * 1000})
    data_extreme = pd.DataFrame({'HRLYEARN': [2000] * 999 + [10_000_000]})
    np.random.seed(42)
    results_normal = [avg_wages(data_normal, epsilon=10.0) for _ in range(200)]
    np.random.seed(42)
    results_extreme = [avg_wages(data_extreme, epsilon=10.0) for _ in range(200)]
    # With clipping, means should be close; without clipping, extreme would be much larger
    assert abs(np.mean(results_normal) - np.mean(results_extreme)) < 5


def test_avg_wages_mean_close_to_true():
    """Over many runs, noisy average should be close to the true average."""
    true_wage_cents = 2500  # $25.00/hr
    data = pd.DataFrame({'HRLYEARN': [true_wage_cents] * 1000})
    np.random.seed(42)
    results = [avg_wages(data, epsilon=1.0) for _ in range(500)]
    np.testing.assert_allclose(np.mean(results), 25.0, atol=0.5)


def test_avg_wages_larger_epsilon_less_noise():
    """Larger epsilon should produce results closer to the true value."""
    data = pd.DataFrame({'HRLYEARN': [2000.0] * 500})
    np.random.seed(42)
    results_low_eps  = [avg_wages(data, epsilon=0.1) for _ in range(300)]
    results_high_eps = [avg_wages(data, epsilon=10.0) for _ in range(300)]
    assert np.var(results_low_eps) > np.var(results_high_eps)

# question 2
def hrs_cdf(lfs):
  a = lfs['ATOTHRS']
  return [len(a[a < i]) for i in range(990)]


def hrs_cdf_dp_laplace(lfs, epsilon):
    # TODO: your code here
    raise NotImplementedError()


def hrs_cdf_dp_gauss(lfs, epsilon, delta):
    # TODO: your code here
    raise NotImplementedError()


def hrs_cdf_v2(lfs, epsilon):
    # TODO: your code here
    raise NotImplementedError()


def rdp_mech(alpha):
    # TODO: your code here
    raise NotImplementedError()


def convert_RDP_ED(alpha, epsilon_bar, delta):
    # TODO: your code here
    raise NotImplementedError()


def encode_response_sales(response, alpha):
    # TODO: your code here
    raise NotImplementedError()


def decode_responses_sales(responses, alpha):
    # TODO: your code here
    raise NotImplementedError()


if __name__ == "__main__":
    # TODO: your code here
    raise NotImplementedError()
