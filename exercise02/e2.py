# Load the data and libraries
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pytest
plt.style.use('seaborn-v0_8-whitegrid')

# ====== question 1 ======
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


# ====== question 2 ======
def hrs_cdf(lfs):
  a = lfs['ATOTHRS']
  return [len(a[a < i]) for i in range(990)]


def hrs_cdf_dp_laplace(lfs, epsilon):
    counts = np.array(hrs_cdf(lfs))
    l1_sensitivity = 989    # adding/removing one person changes at most 989 counts by 1
    return laplace_mech(counts, l1_sensitivity, epsilon)


def hrs_cdf_dp_gauss(lfs, epsilon, delta):
    counts = np.array(hrs_cdf(lfs))
    # L2 sensitivity = sqrt(989): L2 norm of the worst-case change vector
    l2_sensitivity = np.sqrt(989)
    # Gaussian noise scale satisfying (epsilon, delta)-DP
    sigma = l2_sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    noise = np.random.normal(loc=0, scale=sigma, size=np.shape(counts))
    return counts + noise


def hrs_cdf_v2(lfs):
    # Divide by 10 to convert raw ATOTHRS values to actual hours
    a = lfs['ATOTHRS'] / 10.0
    # Range covers 0–168 hours (max possible hours in a week: 7 * 24)
    return [len(a[a < i]) for i in range(169)]


# ====== Question 3 ======
def rdp_mech(alpha):
    lfs = pd.read_csv('2025-02-CSV/pub0225.csv')
    counts = np.array(hrs_cdf_v2(lfs))
    epsilon_bar = 0.001
    l2_sensitivity = np.sqrt(168)
    sigma = l2_sensitivity * np.sqrt(alpha / (2 * epsilon_bar))
    noise = np.random.normal(0, sigma, counts.shape)
    return counts + noise


def convert_RDP_ED(alpha, epsilon_bar, delta):
    return epsilon_bar + np.log(1 / delta) / (alpha - 1)


# ====== Question 4 ======
def encode_response_sales(response, alpha):
   if np.random.random() < alpha:
       return np.random.randint(0, 2) # random 0 or 1
   return response


def decode_responses_sales(responses, alpha):
    n = len(responses)
    p_hat = np.sum(responses) / n   # observed proportion of "True"
    p_true = (p_hat - alpha / 2) / (1 - alpha)
    return p_true * n


# ============= test =============
# Q1 
def test_laplace_mech_scalar_shape():
    """Output shape matches scalar input."""
    result = laplace_mech(5.0, sensitivity=1.0, epsilon=1.0)
    assert np.shape(result) == ()


def test_laplace_mech_vector_shape():
    """Output shape matches vector input."""
    v = np.array([1.0, 2.0, 3.0])
    result = laplace_mech(v, sensitivity=1.0, epsilon=1.0)
    assert result.shape == v.shape


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

# Q2
def _make_lfs(n=500):
    """Helper: synthetic LFS-like DataFrame for testing."""
    np.random.seed(0)
    return pd.DataFrame({'ATOTHRS': np.random.randint(0, 500, size=n)})


def test_hrs_cdf_output_length():
    """hrs_cdf returns a vector of length 990."""
    lfs = _make_lfs()
    assert len(hrs_cdf(lfs)) == 990


def test_hrs_cdf_monotone():
    """CDF must be non-decreasing."""
    lfs = _make_lfs()
    cdf = hrs_cdf(lfs)
    assert all(cdf[i] <= cdf[i+1] for i in range(len(cdf)-1))


def test_hrs_cdf_dp_laplace_output_length():
    """Laplace DP CDF returns a vector of length 990."""
    lfs = _make_lfs()
    result = hrs_cdf_dp_laplace(lfs, epsilon=1.0)
    assert len(result) == 990

def test_hrs_cdf_dp_gauss_output_length():
    """Gaussian DP CDF returns a vector of length 990."""
    lfs = _make_lfs()
    result = hrs_cdf_dp_gauss(lfs, epsilon=1.0, delta=1e-6)
    assert len(result) == 990


def test_hrs_cdf_dp_gauss_mean_close():
    """Mean of many Gaussian DP runs should be close to the true CDF."""
    lfs = _make_lfs(1000)
    true_cdf = np.array(hrs_cdf(lfs))
    np.random.seed(42)
    results = np.array([hrs_cdf_dp_gauss(lfs, epsilon=1.0, delta=1e-6) for _ in range(200)])
    np.testing.assert_allclose(results.mean(axis=0), true_cdf, atol=50)


def test_gauss_less_noise_than_laplace():
    """Gaussian mechanism should have lower variance than Laplace (same epsilon)."""
    lfs = _make_lfs(1000)
    np.random.seed(42)
    lap_results = np.array([hrs_cdf_dp_laplace(lfs, epsilon=1.0) for _ in range(100)])
    gau_results = np.array([hrs_cdf_dp_gauss(lfs, epsilon=1.0, delta=1e-6) for _ in range(100)])
    assert lap_results.var() > gau_results.var()


def test_hrs_cdf_v2_output_length():
    """hrs_cdf_v2 returns a vector of length 169 (0–168 hours)."""
    lfs = _make_lfs()
    assert len(hrs_cdf_v2(lfs)) == 169


def test_hrs_cdf_v2_monotone():
    """hrs_cdf_v2 CDF must be non-decreasing."""
    lfs = _make_lfs()
    cdf = hrs_cdf_v2(lfs)
    assert all(cdf[i] <= cdf[i+1] for i in range(len(cdf)-1))


def test_hrs_cdf_v2_unit_conversion():
    """A person with ATOTHRS=100 (=10.0 hrs) should appear in v2 CDF at i>10."""
    lfs = pd.DataFrame({'ATOTHRS': [100]})  # 10.0 hours
    cdf = hrs_cdf_v2(lfs)
    assert cdf[10] == 0   # i=10: count(hours < 10) → not included
    assert cdf[11] == 1   # i=11: count(hours < 11) → included

# Q3
def test_rdp_mech_output_length():
    """RDP mechanism returns a vector of length 169 (same as hrs_cdf_v2)."""
    result = rdp_mech(alpha=5)
    assert len(result) == 169


def test_rdp_mech_larger_alpha_more_noise():
    """Larger alpha increases sigma, producing higher variance output."""
    np.random.seed(42)
    results_small = np.array([rdp_mech(alpha=2) for _ in range(50)])
    np.random.seed(42)
    results_large = np.array([rdp_mech(alpha=20) for _ in range(50)])
    assert results_large.var() > results_small.var()


def test_convert_RDP_ED_formula():
    """Check conversion formula against manually computed value."""
    # epsilon = epsilon_bar + log(1/delta) / (alpha - 1)
    # = 0.001 + log(1e5) / 4 ≈ 2.8794
    result = convert_RDP_ED(alpha=5, epsilon_bar=0.001, delta=1e-5)
    np.testing.assert_allclose(result, 0.001 + np.log(1e5) / 4, rtol=1e-6)


def test_convert_RDP_ED_larger_delta_smaller_epsilon():
    """Larger delta (weaker requirement) should give smaller epsilon."""
    eps_small_delta = convert_RDP_ED(5, 0.001, delta=1e-6)
    eps_large_delta = convert_RDP_ED(5, 0.001, delta=1e-3)
    assert eps_small_delta > eps_large_delta


def test_convert_RDP_ED_larger_alpha_smaller_epsilon():
    """Larger alpha gives smaller epsilon for the same epsilon_bar and delta."""
    eps_small_alpha = convert_RDP_ED(alpha=2,  epsilon_bar=0.001, delta=1e-5)
    eps_large_alpha = convert_RDP_ED(alpha=20, epsilon_bar=0.001, delta=1e-5)
    assert eps_small_alpha > eps_large_alpha

# Q4
def test_encode_response_true_stays_true():
    """With alpha=0, response is always returned truthfully."""
    assert encode_response_sales(1, alpha=0) == 1
    assert encode_response_sales(0, alpha=0) == 0


def test_encode_response_output_is_binary():
    """Encoded response must always be 0 or 1."""
    for _ in range(100):
        assert encode_response_sales(1, alpha=0.05) in (0, 1)
        assert encode_response_sales(0, alpha=0.05) in (0, 1)


def test_encode_response_alpha1_is_random():
    """With alpha=1, all responses are random — true value has no effect."""
    np.random.seed(42)
    results_true  = [encode_response_sales(1, alpha=1.0) for _ in range(1000)]
    results_false = [encode_response_sales(0, alpha=1.0) for _ in range(1000)]
    # Both should average around 0.5
    np.testing.assert_allclose(np.mean(results_true),  0.5, atol=0.05)
    np.testing.assert_allclose(np.mean(results_false), 0.5, atol=0.05)


def test_decode_recovers_true_count():
    """Decoding many encoded responses should recover the true count."""
    np.random.seed(42)
    n, true_count = 10000, 3000
    responses = [1] * true_count + [0] * (n - true_count)
    encoded = [encode_response_sales(r, alpha=0.05) for r in responses]
    estimated = decode_responses_sales(encoded, alpha=0.05)
    np.testing.assert_allclose(estimated, true_count, rtol=0.05)


def test_decode_all_zeros():
    """If everyone answers 0, estimated count should be close to 0."""
    np.random.seed(42)
    encoded = [encode_response_sales(0, alpha=0.05) for _ in range(5000)]
    estimated = decode_responses_sales(encoded, alpha=0.05)
    np.testing.assert_allclose(estimated, 0, atol=100)


def test_decode_all_ones():
    """If everyone answers 1, estimated count should be close to n."""
    np.random.seed(42)
    n = 5000
    encoded = [encode_response_sales(1, alpha=0.05) for _ in range(n)]
    estimated = decode_responses_sales(encoded, alpha=0.05)
    np.testing.assert_allclose(estimated, n, rtol=0.05)


if __name__ == "__main__":
    lfs = pd.read_csv('2025-02-CSV/pub0225.csv')
    
    # Question 1
    print("=== Q1: Average Wages ===")
    print(f"DP average wage: ${avg_wages(lfs, epsilon=1.0):.2f}/hr")

    # Question 2
    print("\n=== Q2: Hours Worked ===")
    v1_cdf = hrs_cdf(lfs)
    dp_laplace = hrs_cdf_dp_laplace(lfs, epsilon=1.0)
    dp_gauss = hrs_cdf_dp_gauss(lfs, epsilon=1.0, delta=1e-6)
    v2_cdf = hrs_cdf_v2(lfs)

    print(f"\nv1 CDF (first 10):\n {v1_cdf[:10]}")
    print(f"\nDP Laplace (first 10):\n {dp_laplace[:10]}")
    print(f"\nDP Gauss (first 10):\n {dp_gauss[:10]}")
    print(f"\nv2 CDF (first 10, hours):\n {v2_cdf[:10]}")

    # Question 3
    print("\n=== Q3: Renyi Differential Privacy ===")
    alpha, epsilon_bar, delta = 5, 0.001, 1e-5
    rdp_cdf = rdp_mech(alpha)
    epsilon_ed = convert_RDP_ED(alpha, epsilon_bar, delta)
    print(f"RDP noisy CDF (first 10):\n {rdp_cdf[:10]}")
    print(f"Total RDP cost: ({alpha}, {epsilon_bar})-RDP")
    print(f"Converted (epsilon, delta)-DP: epsilon={epsilon_ed:.6f}, delta={delta}")

    # Question 4
    print("\n=== Q4: Randomized Response ===")
    # NOC_43 = 12: Professional occupations in engineering
    ture_responses = (lfs['NOC_43'] == 12).astype(int).tolist()
    true_count = sum(ture_responses)
    encode = [encode_response_sales(r, alpha = 0.05) for r in ture_responses]
    decode = decode_responses_sales(encode, alpha = 0.05)
    print(f"True count:     {true_count}")
    print(f"Estimate count: {decode:.1f}")
    print(f"Error:         {abs(true_count - decode):.1f}({abs(true_count - decode) / true_count * 100:.2f}%)")