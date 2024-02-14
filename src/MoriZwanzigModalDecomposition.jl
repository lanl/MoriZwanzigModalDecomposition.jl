module MoriZwanzigModalDecomposition
using Plots, LaTeXStrings, LinearAlgebra, NPZ
using Statistics, Measures

#MZMD package:

#data-driven mz algorithm as developed in https://epubs.siam.org/doi/abs/10.1137/21M1401759
include("main_mz_algorithm.jl")
export svd_method_of_snapshots, obtain_C, obtain_ker, obtain_future_state_prediction

#extracting modes as developed in https://arxiv.org/abs/2311.09524
include("mzmd_modes.jl")
export compute_markovian_modes, form_companion, mzmd_modes, mzmd_modes_reduced_amps,
mzmd_modes_full_amps, compute_prediction_modes

include("plot_functions.jl")
export plot_field, plotting_mem_norm, plot_mzmd_Mmodes, plot_mzmd_modes, plot_amplitude_vs_frequency_markovian,
plot_amplitude_vs_frequency_mzmd, plot_field_diff, plot_field_diff_mse, plot_c_spectrum


# include("extra_file.jl")
# export my_f

end
