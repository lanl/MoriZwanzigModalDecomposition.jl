# Load necessary libraries
using FFTW, Plots, NPZ, Statistics, LaTeXStrings
include("plot_functions.jl")

# Function to compute the vortex shedding frequency using FFT
function vortex_shedding_frequency(data, dt)
    # Perform FFT on the time-series data
    N = length(data)  # Number of data points
    freq_data = fft(data)
    
    # Compute the corresponding frequencies
    freqs = (0:N-1) ./ (N * dt)  # Frequency axis

    # Compute the power spectrum (magnitude of FFT)
    power_spectrum = abs.(freq_data[1:div(N,2)])  # Keep positive frequencies only
    freqs = freqs[1:div(N,2)]  # Positive frequencies

    # Find the index of the maximum peak in the power spectrum
    peak_idx = argmax(power_spectrum)
    shedding_freq = freqs[peak_idx]

    # Plot power spectrum for visualization
    gr(size=(570,450), xtickfontsize=12, ytickfontsize=12, xguidefontsize=20, yguidefontsize=14, legendfontsize=12,
    dpi=300, grid=(:y, :gray, :solid, 1, 0.4), palette=cgrad(:plasma, 3, categorical = true));
    plt = plot(freqs, power_spectrum/maximum(power_spectrum), xlabel="Frequency (Hz)", xlims=(0, 1), ylabel="Normalized power", title="Power Spectrum of Vortex Shedding")
    display(plt)
    return shedding_freq
end

# Example usage:
# Assume `vorticity_data` is a time-series array of vorticity or velocity measured at a point downstream of the cylinder
# and `dt` is the time step of the data.

# Generate sample data (replace with actual vorticity/velocity data)
# vort_path = "/Users/l352947/mori_zwanzig/mori_zwanzig/mzmd_code_release/data/vort_all_re600_t5100.npy"
re_num = 200;
vort_path = "../data/vort_all_re200_t145.npy"
X = npzread(vort_path);
X = X .- mean(X, dims=2);

T = size(X, 2);
t_skip = 1;
X = X[:, 1:t_skip:end] .- mean(X, dims=2);
dt = t_skip*0.2; #physical time dt
t = 0:dt:((T-1)*dt);  # Time vector

Y = copy(X);
F_select = reshape(Y, (199, 449, T))[100, 200, :]; #select a point in wake to compute psd

# simple sanity check
# f_shedding = 2.0  # Example shedding frequency (Hz) for demonstration
# vorticity_data = sin.(2 * π * f_shedding .* t)  # Simulated vorticity signal with shedding frequency

vorticity_data = F_select[1:end];
plt = plot(vorticity_data[1:100])
display(plt)

# Call the function to compute the shedding frequency
shedding_frequency = vortex_shedding_frequency(vorticity_data, dt)
println("Estimated vortex shedding frequency: $shedding_frequency Hz")


