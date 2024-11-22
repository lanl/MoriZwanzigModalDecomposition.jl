using Plots, LaTeXStrings, LinearAlgebra, NPZ
using Statistics


#--------- Include files
include("plot_functions.jl")
include("./src/compute_obervability_amplitudes_optimized.jl")
include("./src/main_mz_algorithm.jl")
include("./src/mzmd_modes.jl")


#--------- Load data
re_num = 9;
method="companion"
# a_method = "lsa";
a_method = "x0";
# sample = parse(Int, ARGS[1]);
sample = 1;
println("running e.method $(method) and amp.method $(a_method) eigendecomp computing errors over r, k, d")

#--------- Load data
t_skip = 3;
dt = t_skip*0.2;
# vort_path = "/Users/l352947/mori_zwanzig/modal_analysis_mzmd/mori_zwanzig/mzmd_code_release/data/vort_all_re600_t5100.npy"
vort_path = "/Users/l352947/mori_zwanzig/mori_zwanzig/mzmd_code_release/data/vort_all_re600_t5100.npy"
X = npzread(vort_path)[:, 1100:5000];

# gx = load_grid_file(nx)
m = size(X, 1);
t_end = 1200 #total n steps is 751
t_pred = 120
T_train = ((sample):(t_end+sample-1))[1:t_skip:end];
T_test = ((t_end+sample):(t_end + t_pred + sample))[1:t_skip:end];
time_train = dt * T_train
time_test = dt * t_skip * T_test;
t_all = 0:dt:(dt*(t_end + t_pred +1));
X_train = X[:,T_train];
X_test = X[:,T_test];


t_ = 20
t_r = round((dt*(t_-2)), digits=2)

plt = plot_field(X, t_, "DNS ~ ~ t = $(t_r)s")
savefig(plt, "figures/dns_snapshot_$(t_).png")

#-------- Set params
#number of memory terms:
n_ks = 21
#number of snapshots used in fitting
t_win = size(T_train, 1) - n_ks - 1;
r = 30;


#-------- Compute MZ on POD basis observables
X1 = X_train[:, 1:t_win]; #Used to obtain Ur on 1:twin
X0 = X_train[:, 1];

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
    println("----------------------------");
    println("Two time covariance matrices: size(C)) = ", size(Cov));
    println("Markovian operator: size(M)  = ", size(M));
    println("All mz operators: size(Ω)  = ", size(Ω));
    return Cov, M, Ω
end
Cov, M, Ω = obtain_mz_operators()
plotting_mem_norm(Ω)

function obtain_omega_norms(Ω)
    Ω_norms = zeros(n_ks)
    r_ = size(Ω, 2);
    I_ = I(r_);
    for i in 1:n_ks
        if i==1
            Ω_norms[i] = norm(Ω[i,:,:] - I_)
        else
            Ω_norms[i] = norm(Ω[i,:,:])
        end
    end
    return Ω_norms
end
Ω_norms = obtain_omega_norms(Ω);
# npzwrite("./omega_norms_r$(r)_k$(n_ksk).npy", Ω_norms)

plt = plot(Ω_norms, yaxis=:log)
display(plt)

# C = form_companion(Ω, r, n_ks);

#----------------------------------------------------------------
                        #future state predictions
#----------------------------------------------------------------

function simulate_modes(Ur, evects, a, r, λ, dt, t)
    a_length = length(a);
    t_length = length(t);
    time_dynamics = complex(zeros(a_length, t_length));
    ts = (0:t_length-1) * dt;
    omega = log.(λ)/dt;
    for i in 1 : t_length
        time_dynamics[:,i] = (a.*exp.(ts[i] .* omega));
    end
    X_pred = evects[1:r,:] * time_dynamics;
    Xmz_pred = Ur * X_pred;
    return real.(Xmz_pred)
end

function predict_mzmd(amp_method, t_all, X_proj, C, Ur, r, n_ks)
    if amp_method=="x0"
        Λc, a_mz, Vc = mzmd_modes_reduced_amps_pred(C, X, Ur, r, n_ks);
        solution_modes = simulate_modes(Ur, Vc, a_mz, r, Λc, dt, t_all);
    end
    if amp_method=="lsa"
        Λc, Vc = eigen(C);
        e_vects = Vc[1:r,:];
        a, amps, Q = compute_amplitudes_observability_matrix(e_vects, Λc, X_proj, Ur, m);
        solution_modes = simulate_modes(Ur, Q, a, r, Λc, dt, t_all);
    end
    return Λc, Vc, a, amps, Q, solution_modes
end

function predict_dmd(amp_method, t_all, X0r, M, Ur, r)
    if amp_method=="x0"
        lambda, e_vects, a, amps = compute_markovian_modes_reduced(M, X0r[:, 1], Ur, r)
        solution_modes = simulate_modes(Ur, e_vects, a, r, lambda, dt, t_all);
    end
    if amp_method=="lsa"
        lambda, e_vects = eigen(M);
        a, amps, Q = compute_amplitudes_observability_matrix(e_vects, Λ, X_proj, Ur, m);
        solution_modes = simulate_modes(Ur, Q, a, r, Λ, dt, t_all);
    end
    return lambda, e_vects, a, amps, solution_modes
end

# Λc, Vc, a, amps, Q, solution_modes = predict_mzmd("lsa", 1:751, X_proj, C, Ur, r, n_ks);
# lambda, e_vects, a_m, amps_m, X_dmd = predict_dmd("x0", 1:751, X0r, M, Ur, r);

# #plot spectrum
# # plt = plot_amplitude_vs_frequency_select_amps(lambda, amps_m, dt, "dmd")
# # display(plt)

# # plot_c_spectrum(lambda, "dmd")

# #compute modes and spectrum of MZMD
# Λc_red, Φ_mz, a_mz_red, amp_mz_red, Vc = mzmd_modes_reduced_amps(C, X, Ur, r, n_ks);
# amp_norm = zeros(r*n_ks);
# for i in 1:r*n_ks
#     amp_norm[i] = norm(Φ_mz[:,i]*amps[i]);
# end
# plt = plot_amplitude_vs_frequency_select_amps(Λc_red, norm(Q)*sqrt(m)*amp_norm/10, dt, "mzmd")
# display(plt)
# plot_c_spectrum(Λc, "mzmd")


# function compute_avg_pressure_sig(X)
#     F = reshape(X, (nx, nz, T)); #F[:,:,t]
#     F_avgz = mean(F, dims=2);
#     p_sig = mean(F_avgz[:,1,:], dims=2);
#     return p_sig[:, 1]
# end
# p_amp_dns = compute_avg_pressure_sig(X .+ X_mean);
# p_amp_mz = compute_avg_pressure_sig(solution_modes .+X_mean);
# p_amp_dmd = compute_avg_pressure_sig(X_dmd .+ X_mean);


# function plot_p_amps()
#     gr(size=(900, 900))
#     plt = plot(p_amp_dns, color="black", label="DNS")
#     plot!(p_amp_mz, color="blue", label="MZMD")
#     plot!(p_amp_dmd, color="red", label="DMD")
#     display(plt)
# end
# plot_p_amps()

# sorted_indices_mz = sortperm(abs.(amp_norm), rev=true);
# for i in 1:2:15
#     i_mode = sorted_indices_mz[i]
#     plt_m = plot_mzmd_modes_axisymmetric(Φ_mz, nx, nz, gx[1:nx], i_mode)
#     # savefig(plt_m, "./figures/mzmd_dominant_modes_r$(r)_k$(n_ks)_i$(i_mode).png")
#     display(plt_m)
# end


function save_dominant_modes()
    sorted_indices_mz = sortperm(abs.(amp_norm), rev=true);
    for i in 1:2:15
        i_mode = sorted_indices_mz[i]
        #plot mode check:
        # plt_m = plot_mzmd_modes_axisymmetric(Φ_mz, nx, nz, gx[1:nx], i_mode)
        # savefig(plt_m, "./figures/mzmd_dominant_modes_r$(r)_k$(n_ks)_i$(i_mode).png")
        # display(plt_m)
        Φ = reshape(norm(Q)*sqrt(m)*amp_norm[i_mode]/10 * Φ_mz[:, i_mode]/norm(Φ_mz[:, i_mode]), (nx, nz)); 
        println("i = ", i_mode, " amp = ", norm(Q)*sqrt(m)*amp_norm[i_mode]/10)
        Φ_full = permutedims(hcat(Φ, reverse(Φ, dims=2)), (2, 1));
        npzwrite("./modes/phi_mz_mode_i$(i)_r$(r)_k$(n_ks).npy", Φ_full)
    end
end
# save_dominant_modes()




# #plot diff fields
function obtain_save_diffs()
    diff_mz = mean((solution_modes[:, 1:701] .- X[:, 1:701]).^2, dims=2);
    diff_dmd = mean((X_dmd[:, 1:701] .- X[:, 1:701]).^2, dims=2);
    c_max = maximum(diff_mz/mean(X.^2))

    dmd_diff = reshape(diff_dmd, (nx, nz)); 
    dmd_diff = permutedims(hcat(dmd_diff, reverse(dmd_diff, dims=2)), (2, 1));

    mz_diff = reshape(diff_mz, (nx, nz)); 
    mz_diff = permutedims(hcat(mz_diff, reverse(mz_diff, dims=2)), (2, 1));

    npzwrite("./diff_dmd.npy", dmd_diff/mean(X.^2))
    npzwrite("./diff_mz.npy", mz_diff/mean(X.^2))

    plt = plot_diff_field(diff_dmd/mean(X.^2), nx, nz, gx[1:nx], c_max, "DMD ~ Pointwise ~ MSE")
    display(plt)
end


function obtain_save_preds()
    F2 = reshape(X[:, 751] .+ X_mean, (nx, nz)); 
    F2 = permutedims(hcat(F2, reverse(F2, dims=2)), (2, 1));
    npzwrite("./dns_tf.npy", F2)

    Fmz = reshape(solution_modes[:, 751] .+ X_mean, (nx, nz)); 
    Fmz = permutedims(hcat(Fmz, reverse(Fmz, dims=2)), (2, 1));
    npzwrite("./mz_tf.npy", Fmz)

    Fdmd = reshape(X_dmd[:, 751] .+ X_mean, (nx, nz)); 
    Fdmd = permutedims(hcat(Fdmd, reverse(Fdmd, dims=2)), (2, 1));
    npzwrite("./dmd_tf.npy", Fdmd)
end
# obtain_save_preds()


# savefig(plt, "./figures/dmd_diff_field_r$(r).png")

# plt = plot_diff_field(diff_mz/mean(X.^2), nx, nz, gx[1:nx], c_max, "MZMD ~ Pointwise ~ MSE")
# display(plt)
# savefig(plt, "./figures/mzmd_diff_field_r$(r)_k$(n_ks).png")

# println("total error mzmd = ", mean(diff_mz)/mean(X.^2))
# println("total error dmd = ", mean(diff_dmd)/mean(X.^2))


#simulating predictions
# out_path = "./prediction_compare_all_full_r$(r)_k$(n_ks).mov"
# animate_field_comparison(X .+ X_mean, solution_modes .+ X_mean, X_dmd .+ X_mean, nx, nz, gx[1:nx], out_path)


