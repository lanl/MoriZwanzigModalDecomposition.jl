using Plots, LaTeXStrings, LinearAlgebra, NPZ
using Statistics
using StatsBase

#--------- Include files
include("plot_functions.jl")
include("./src/compute_obervability_amplitudes_optimized.jl")
include("./src/main_mz_algorithm.jl")
include("./src/mzmd_modes.jl")


#--------- Load data
re_num = 200;
method="companion"
# a_method = "lsa";
a_method = "x0";
# sample = parse(Int, ARGS[1]);
sample = 1;
println("running e.method $(method) and amp.method $(a_method) eigendecomp computing errors over r, k, d")

#--------- Load data
t_skip = 1;
dt = t_skip*0.2;
# vort_path = "/Users/l352947/mori_zwanzig/modal_analysis_mzmd/mori_zwanzig/mzmd_code_release/data/vort_all_re600_t5100.npy"
vort_path = "../data/vort_all_re200_t145.npy"
X = npzread(vort_path) #T=1500
X_mean = mean(X, dims=2);
X = X .- X_mean;
m = size(X, 1);

# gx = load_grid_file(nx)
m = size(X, 1);
t_end = 110 #total n steps is 751
t_pred = 30;
T_train = ((sample):(t_end+sample-1))[1:t_skip:end];
T_test = ((t_end+sample):(t_end + t_pred + sample))[1:t_skip:end];
time_train = dt * T_train
time_test = dt * t_skip * T_test;
t_all = 0:dt:(dt*(t_end + t_pred +1));
X_train = X[:,T_train];
X_test = X[:,T_test];
X = nothing;


#-------- Set params
#number of memory terms:
# n_ks = 21
n_ks = 2;
#number of snapshots used in fitting
t_win = size(T_train, 1) - n_ks - 1;
r = 20;


#-------- Compute MZ on POD basis observables
X1 = X_train[:, 1:t_win]; #Used to obtain Ur on 1:twin
X0 = X_train[:, 1];

S, Ur, X_proj = svd_method_of_snapshots(X_train, r, subtractmean=true)

#initial condtions with memory for obtaining predictions
X0r = X_proj[:, 1:n_ks];

function obtain_mz_operators()
    #compute two time covariance matrices
    Cov = obtain_C(X_proj, t_win, n_ks);
    #compute MZ operators with Mori projection
    M, Ω = obtain_ker(Cov, n_ks);
    println("----------------------------");
    println("Two time covariance matrices: size(C)) = ", size(Cov));
    println("Markovian operator: size(M)  = ", size(M));
    println("All mz operators: size(Ω)  = ", size(Ω));
    return Cov, M, Ω
end
Cov, M, Ω = obtain_mz_operators()
# plotting_mem_norm(Ω);


function pointwise_mse_errors(t1, gt_data, Xmz_pred)
    mz_pred = (Xmz_pred .+ X_mean)[:, t1];
    # mz_err = mean((mz_pred .- gt_data).^2, dims=2)/maximum(gt_data);
    mz_err = mean((mz_pred .- gt_data).^2, dims=2)/maximum(X_mean);
    max_mz = maximum(mz_pred .- gt_data, dims=2);
    return mz_err, max_mz
end

function dmd_predictor(x0, A, T_pred)
    dmd_pred = zeros(size(x0,1), T_pred)
    for t in 1:T_pred
        dmd_pred[:, t] = A^(t)*x0;
    end
    return dmd_pred
end


mz_pred = obtain_future_state_prediction(Ur, Ur'*X_test[:, 1:n_ks], Ω, n_ks, t_pred)
dmd_pred = obtain_future_state_prediction_dmd(Ur, Ur'*X_test[:, 1], Ω[1, :, :], 1, t_pred)
prediction_mz = real.(Ur*mz_pred);
prediction_dmd = real.(Ur*dmd_pred);


Fdmd = reshape((prediction_dmd), (199, 449, size(prediction_dmd,2)));
Fmz = reshape((prediction_mz), (199, 449, size(prediction_mz,2)));
F = reshape((X_test), (199, 449, size(X_test,2)));

function probability_distribution(data, bins::Int = 100)
    hist = fit(Histogram, data, nbins = bins)
    norm_h = norm(hist)
    probabilities = hist.weights ./ norm_h
    return hist.edges[1], probabilities
end

function plot_pdf(data1, data2, data3, n_bins, n_ks, r)
	gr(size=(570,450), xtickfontsize=12, ytickfontsize=12, xguidefontsize=20, yguidefontsize=14, legendfontsize=12,
		dpi=200, grid=(:y, :gray, :solid, 1, 1), palette=cgrad(:plasma, 3, categorical = true));

    # Calculate probability distribution
    edges1, probabilities1 = probability_distribution(data1, n_bins)
    edges2, probabilities2 = probability_distribution(data2, n_bins)
    edges3, probabilities3 = probability_distribution(data3, n_bins)

    # Midpoints of bins for plotting
    bin_centers1 = [(edges1[i] + edges1[i+1]) / 2 for i in 1:length(edges1)-1]
    bin_centers2 = [(edges2[i] + edges2[i+1]) / 2 for i in 1:length(edges2)-1]
    bin_centers3 = [(edges3[i] + edges3[i+1]) / 2 for i in 1:length(edges3)-1]

    xlim_min = 1.2*minimum(bin_centers1);
    xlim_max = 1.2*maximum(bin_centers1);

    # Plot the probability distribution
    plt = plot(bin_centers1, probabilities1 .+1e-18, yaxis=:log, seriestype = :line, lw=3, 
        label=L"\textrm{DNS}", color="purple")
    plot!(bin_centers3, probabilities3 .+1e-18, yaxis=:log, seriestype = :line, lw=2,
        legend = true, color="green4", label=L"\textrm{DMD}", linestyle=:dot)
    plot!(bin_centers2, probabilities2 .+1e-18,  yscale=:log10, minorgrid=true,
         seriestype = :line, lw=1.5,
         color="blue", label=L"\textrm{MZMD}", linestyle=:dash, ms=:x, legendfontsize=10, 
         grid = true, xlim=(xlim_min, xlim_max), ylim=(1e-6, 2))
    yticks = [10.0^i for i in -5:1:2]  # Set y-axis ticks for each decade
    yticks!(yticks)
    # kl1val = round(kl1, digits=3);
    # kl2val = round(kl2, digits=3);
    # annotate!([-0.05], [3.5e-4], 
    # text(L"D_{KL}(p_{dns} || p_{mzmd}) = %$kl1val", 10, :black, :left))
    # annotate!([-0.05], [1e-4], 
    #     text(L"D_{KL}(p_{dns} || p_{hodmd}) = %$kl2val", 10, :black, :left))

    title!(L"\textrm{Long ~ time ~ statistics}", titlefont=16)
    xlabel!(L"vorticity", xtickfontsize=10, xguidefontsize=14)
    ylabel!(L"pdf", ytickfontsize=10, yguidefontsize=14)
    savefig(plt, "./figures/vorticity_pdf_long_time_comparison_k$(n_ks)_r$(r).pdf")
    display(plt)
    # return probabilities1, probabilities2, probabilities3, probabilities4, plt
end

t_ = 1:t_pred;
nbins_ = 130;
plot_pdf(vec(F[:,:,t_]), vec(Fmz[:,:,t_]), vec(Fdmd[:,:,t_]), nbins_, n_ks, r)
# plt = plot_pdf(vec(F[:,:,t_]), vec(Fmz[:,:,t_]), vec(Fdmd[:,:,t_]), nbins_)
# display(plt)

#note you can seperate the flow into different regions and look at the statistics of each
#region of downstream direction in the following:
# x = x1:x2; #e.g. x1 = 1; x2 = 500; 
# plt = plot_pdf3(vec(F[x,:,t_1]), vec(Fmz[x,:,t_]), vec(Fdmd[x,:,t_]), 130)
# display(plt)
