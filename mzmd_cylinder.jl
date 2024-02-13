using MZMD
using Plots, LaTeXStrings, LinearAlgebra, NPZ
using Statistics, Measures

#This file demonstrates the MZMD approach on the flow over a cyclinder
    # - Extracts and plots mzmd modes, decaying memory effects, and pointwise errors


#--------- Load data
re_num = 200;
vort_path = "./data/vort_all_re200_t145.npy"
#time window for training
T_train = 1:130
#time window for measuring test set errors
T_test = 131:145

X = npzread(vort_path);
m, T = size(X);

X_train = X[:,T_train];
X_test = X[:,T_test];

#plot final snapshot of vorticity
plot_field(X, T)


#-------- Set params
#number of memory terms:
n_ks = 10
#number of snapshots used in fitting
t_win = size(T_train, 1) - n_ks - 1;
#time step (set from dns data sample rate)
dt = 0.2;
#rank of svd
r = 7;


#-------- Compute MZ on POD basis observables
X1 = X_train[:, 1:t_win]; #Used to obtain Ur on 1:twin
X0 = X_train[:, 1];
X_mean = mean(X_train, dims=2);

#compute svd based observables (X_proj: states projected onto pod modes)
#method of snapshot more efficient for tall skinny X
S, Ur, X_proj = svd_method_of_snapshots(X_train, r, subtractmean=true)

#initial condtions with memory for obtaining predictions
X0r = X_proj[:, 1:n_ks];

function obtain_mz_operators()
    #compute two time covariance matrices
    Cov = obtain_C(X_proj, t_win, n_ks);
    #compute MZ operators with Mori projection
    M, Ω = obtain_ker(Cov, n_ks);
    println("--------- Obtained linear MZ operators -----------");
    println("Two time covariance matrices: size(C)) = ", size(Cov));
    println("Markovian operator: size(M)  = ", size(M));
    println("All mz operators: size(Ω)  = ", size(Ω));
    return Cov, M, Ω
end
Cov, M, Ω = obtain_mz_operators()
#plot the relative contribution of Memory terms wrt Markovian
plotting_mem_norm(Ω, M, n_ks)


#---- MZMD modes and spectrum

#Markovian modes: (equivalent to DMD modes)
lambda, Φ_markovian, a_m = compute_markovian_modes(M, Ur, X0, r)
for i in 1 : 2 : r
    plt_m = plot_mzmd_Mmodes(Φ_markovian, r, i)
    display(plt_m)
end

#Compute and plot modes of block companion matrix
C = form_companion(Ω, r, n_ks);
Λc, Φ_mz, a, amp = mzmd_modes_reduced_amps(X_train, C, Ur, r, n_ks) #fast and accurate algorithm

#plot mzmd modes ranked by svd
for i in 1 : 2 : r
    plt_m = plot_mzmd_modes(Φ_mz, r, n_ks, i)
    display(plt_m)
    savefig(plt_m, "./figures/mzmd_mode_i$(i)_r$(r)_nk$(n_ks).png")
end

#plot spectrum and amplitudes
plt = plot_amplitude_vs_frequency_markovian(lambda, a_m, r, 1)
display(plt)

plt = plot_amplitude_vs_frequency_mzmd(Λc, amp, r, n_ks)
display(plt)
plot_c_spectrum(Λc)



#-------------- Aanalysis of generaization errros

#setting initial condition (with memory in reduced space)
X0_gen = Ur'*X_test[:, 1:n_ks];

function compute_mzmd_prediction_full(T_test)
    X0_gen = Ur'*X_test[:, 1:n_ks];
    t_test = size(T_test,1);
    Xmz_pred_lr = obtain_future_state_prediction(Ur, X0_gen, Ω, n_ks, t_test) #low rank prediction
    println(size(Xmz_pred_lr))
    Xmz_pred = ((Ur * Xmz_pred_lr))[:,1:t_test];
    return Xmz_pred
end

#computing mzmd prediction with all modes
Xmz_pred = compute_mzmd_prediction_full(T_test);

function compute_dmd_prediction_full(T_test)
    X0_gen = Ur'*X_test[:, 1];
    t_test = size(T_test,1);
    Xdmd_pred_lr = obtain_future_state_prediction(Ur, X0_gen, Ω, 1, t_test) #low rank prediction
    println(size(Xdmd_pred_lr))
    Xdmd_pred = ((Ur * Xdmd_pred_lr))[:,1:t_test];
    return Xdmd_pred
end

#computing dmd prediction with all modes
Xdmd_pred = compute_dmd_prediction_full(T_test);

function pointwise_mse_full()
    #compute pointwise mse between prediction and gt
    mz_err = mean(((Xmz_pred .+ X_mean) .- (X_test)).^2, dims=2)
    dmd_err = mean(((Xdmd_pred .+ X_mean) .- (X_test)).^2, dims=2)
    c_max = maximum(dmd_err)
    println("total mse DMD: ", mean(dmd_err))
    println("total mse MZMD: ", mean(mz_err))
    println("relative improvement of MZMD: ", "%", 100*(mean(dmd_err) - mean(mz_err))/mean(dmd_err))
    plot_field_diff_mse(mz_err, dmd_err, size(T_test,1), c_max)
end

#plotting the pointwise prediction errors
pointwise_mse_full()
