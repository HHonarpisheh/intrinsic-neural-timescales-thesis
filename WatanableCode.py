import numpy as np
import matplotlib.pyplot as plt

def TW_AutoCorrFactor01(series, time_resolution, n_lags=30, Q=0, n_stds=2):
    """
    Compute the intrinsic timescale (STS) and the autocorrelation function (ACF) for a given time series.

    Parameters:
        series (array-like): Signal time series (1D array).
        time_resolution (float): Time resolution of the time series (in seconds).
        n_lags (int, optional): Number of lags of the autocorrelation function to compute. Default is 20.
        Q (int, optional): Number of lags beyond which the theoretical ACF is deemed to have died out. Default is 0.
        n_stds (float, optional): Number of standard deviations for confidence bounds. Default is 2.

    Returns:
        STS (float): Signal timescale (sum of positive ACF values times time resolution).
        ACF (array): Autocorrelation function values.
        lags (array): Lags corresponding to ACF values.
        bounds (array): Confidence bounds for the ACF assuming the series is an MA(Q) process.
    """
    # Ensure the series is a 1D array
    series = np.asarray(series).flatten()
    n = len(series)

    if n_lags <= 0 or n_lags >= n:
        raise ValueError("n_lags must be a positive integer less than the length of the series.")
    if Q < 0 or Q >= n_lags:
        raise ValueError("Q must be a non-negative integer less than n_lags.")
    if n_stds < 0:
        raise ValueError("n_stds must be non-negative.")

    # Compute the autocorrelation function (ACF)
    n_fft = 2 ** (np.ceil(np.log2(n)) + 1).astype(int)
    F = np.fft.fft(series - np.mean(series), n=n_fft)
    F = F * np.conj(F)
    acf = np.fft.ifft(F).real[:n_lags + 1]  # Retain only non-negative lags
    acf /= acf[0]  # Normalize

    # Compute confidence bounds
    sigma_q = np.sqrt((1 + 2 * np.sum(acf[1:Q + 1] ** 2)) / n)
    bounds = sigma_q * np.array([n_stds, -n_stds])

    # Calculate lags
    lags = np.arange(n_lags + 1)

    # Compute signal timescale (STS)
    positive_acf_sum = np.sum(acf[acf > 0])
    sts = positive_acf_sum * time_resolution

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.stem(lags, acf, linefmt='r-', markerfmt='ro', basefmt='k-', label='ACF')
    plt.axhline(bounds[0], color='b', linestyle='--', label='Confidence Bound')
    plt.axhline(bounds[1], color='b', linestyle='--')
    plt.axhline(0, color='k', linestyle='-')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title('Autocorrelation Function (ACF)')
    plt.grid(True)
    plt.legend()
    plt.show()

    return sts, acf, lags, bounds

# Example usage
if __name__ == "__main__":
    # Simulated example series
    np.random.seed(42)
    example_series = np.random.randn(500)
    time_res = 0.5

    sts, acf, lags, bounds = TW_AutoCorrFactor01(example_series, time_res)
    print("Signal Timescale (STS):", sts)
    print("Autocorrelation Function (ACF):", acf)
    print("Lags:", lags)
    print("Confidence Bounds:", bounds)
