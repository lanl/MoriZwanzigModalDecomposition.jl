using Plots, LaTeXStrings, LinearAlgebra, NPZ
using Statistics
using Colors, ColorSchemes


""" 
Comparing MZMD errors
    - Measuring reconstrucion and validation errors
"""

include("./src/main_mz_algorithm.jl")
include("./src/compute_obervability_amplitudes_optimized.jl")
include("./plot_functions.jl")
include("./src/mzmd_modes.jl")


re_num = 200;
method="companion"
a_method = "x0"
# sample = parse(Int, ARGS[1]);
sample = 20;
println("  Sample = ", sample)
println("running e.method $(method) and amp.method $(a_method) eigendecomp computing errors over r, k, d")


#--------- Load data
t_skip = 1;
#number of dominant modes to use in prediction
# nmodes_ = 1000;
nmodes_="all";
dt = t_skip*0.2;
vort_path = "../data/vort_all_re200_t145.npy"
X = npzread(vort_path) #T=1500


X_mean = mean(X, dims=2);
X = X .- X_mean;
#check robustness to noise
m = size(X,1);

#training time
t_end = 120 #total n steps is 751
#prediction time
t_pred = 20

T_train = ((1):(t_end+1-1))[1:t_skip:end];
T_test = ((t_end+sample):(t_end + t_pred + sample))[1:t_skip:end];
time_train = dt * T_train
time_test = dt * t_skip * T_test;
t_all = 0:dt:(dt*(t_end + t_pred +1));
X_train = X[:,T_train];
# X_train = X_train .+ 0.15*maximum(X_train) .* randn(size(X_train))
X_test = X[:,T_test];

t_end = round(Int, t_end/t_skip);
t_pred = round(Int, t_pred/t_skip);
println("t end = ", t_end);
U, S, V = svd(X_train);

function mzmd_modes_reduced_amps_for_errors(twin, X, r, n_ks)
    #compute modes and amplitudes from C
    function obtain_mz_operators_(G, t_win, n_ks)
        Cov = obtain_C(G, t_win, n_ks);
        _, Ω = obtain_ker(Cov, n_ks);
        println("----------------------------");
        return Ω
    end
    Ur = U[:, 1:r];
    X_proj = Ur' * X[:, T_train];
        Ω = obtain_mz_operators_(X_proj, twin, n_ks);
        C = form_companion(Ω, r, n_ks);
        Λc, Vc = eigen(C);
        Z0 = Ur'*X_test[:, 1:(n_ks+1)];
        z0 = zeros(n_ks*r)
        for k in 1:n_ks
            # z0[((k-1)*r+1):k*r] = Z0[:,k]
            z0[((k-1)*r+1):k*r] = Z0[:,(n_ks - k + 1)]
        end
        a = pinv(Vc)*z0;
    return Λc, a, Vc[1:r, :], Ur
end

function pointwise_mse_errors(t1, gt_data, Xmz_pred)
    mz_pred = (Xmz_pred .+ X_mean)[:, t1]; 
    # mz_pred = Xmz_pred[:, t1];
    # gt_data = X_test[:, t2];
    mz_err = mean(mean((mz_pred .- gt_data).^2, dims=2))/mean(gt_data.^2);
    max_mz = maximum(mz_pred .- gt_data)/maximum(gt_data)
    return mz_err, max_mz
end

function simulate_modes_time_delay(Ur, evects, a, r, λ, dt, t)
    a_length = length(a);
    t_length = length(t);
    time_dynamics = complex(zeros(a_length, t_length));
    ts = (0:t_length-1) * dt;
    omega = log.(λ)/dt;
    for i in 1 : t_length
        time_dynamics[:,i] = (a.*exp.(ts[i] .* omega));
    end
    X_pred = evects * time_dynamics; #in higher dim delay space
    Xmz_pred = Ur * X_pred[1:r,:];
    return real.(Xmz_pred)
end

function select_dominant_modes(a, Ur, Vc, r, n_ks, lam, num_modes)
    Φ_mz = Ur*Vc[1:r,:];
    amp = zeros(r*n_ks);
    for i in 1:r*n_ks
        amp[i] = norm(Φ_mz[:,i]*a[i]);
    end
    #sort according to largest amplitude:
    ind = sortperm(abs.(amp), rev=true)
    a_dom = a[ind][1:num_modes];
    Vc_dom = Vc[1:r, ind][:, 1:num_modes];
    lam_dom = lam[ind][1:num_modes];
    return a_dom, Vc_dom, lam_dom
end


function compute_error_r_nk(nk_range, nd_range, r_range, nk_end)
    t_pred = length(T_test)
    err_mzmd = zeros(t_pred, size(nk_range,1), size(nd_range,1), size(r_range,1));
    err_mzmd_max = zeros(t_pred, size(nk_range,1), size(nd_range,1), size(r_range,1));
    err_recon_mzmd = zeros(size(nk_range,1), size(nd_range,1), size(r_range,1));
    err_recon_max = zeros(size(nk_range,1), size(nd_range,1), size(r_range,1));
    r_idx = 1;
    for r in r_range
        nd_idx = 1;
        for d in nd_range
            nk_idx = 1;
            for k in nk_range
                t_win = size(T_train, 1) - k - 1;
                # lam, Q, a, Ur = compute_modes_amplitudes_time_delay_observables(X, r, k, d, t_win, method);
                lam, a, Q, Ur = mzmd_modes_reduced_amps_for_errors(t_win, X, r, k)
                if nmodes_=="all"
                    num_modes = k*r;
                else
                    num_modes = minimum([k*r, nmodes_]);
                end
                a_dom, Vc_dom, lam_dom = select_dominant_modes(a, Ur, Q, r, k, lam, num_modes) #dominant three modes
                    solution_modes = simulate_modes_time_delay(Ur, Vc_dom, a_dom, r, lam_dom, dt, time_test)
                    for t in 1:(t_pred-nk_end)
                        err_mzmd[t, nk_idx, nd_idx, r_idx], err_mzmd_max[t, nk_idx, nd_idx, r_idx] = 
                        pointwise_mse_errors(t, X_test[:, t+k-1] .+ X_mean, solution_modes)
                    end
                println(" r = ", r, " nk = ", k, " nd = ", d, " mz err = ", mean(err_mzmd[:, nk_idx, nd_idx, r_idx]),
                " mz err max = ", mean(err_mzmd_max[:, nk_idx, nd_idx, r_idx]), "   recon err = ", err_recon_mzmd[nk_idx, nd_idx, r_idx])
                nk_idx += 1;
            end
            nd_idx += 1;
        end
        r_idx += 1;
    end
    return err_mzmd, err_mzmd_max, err_recon_mzmd, err_recon_max
end

# r_range = cat(3:2:9);
r_range = 10:50:1000;
nk_range = 1:12
nd_range = [1];
nk_end = nk_range[end];

err_mzmd, err_mzmd_max, err_recon_mzmd, err_recon_max = compute_error_r_nk(nk_range, nd_range, r_range, nk_end);

nr = length(r_range);
nk = length(nk_range);
nd = length(nd_range);
nk_end = nk_range[end];
nd_end = nd_range[end];

t_pred_length = t_pred;


function make_directory(path::String)
    if !isdir(path)  # Check if the directory already exists
        mkdir(path)
        println("Directory created at: $path")
    else
        println("Directory already exists at: $path")
    end
end

dir_path = "dom_modes_gen_error_x0_test_dt$(t_skip)_te$(t_end)_re$(re_num)_nm$(nmodes_)"
make_directory(dir_path)

npzwrite("$(dir_path)/cyl_ax0_err_mzmd_delay_k1dmd_dts$(t_skip)_st$(sample)_$(method)_$(a_method)_re$(re_num)_modes_nr$(nr)_nk$(nk)_tp$(t_pred_length)_te$(t_end)_nke$(nk_end)_nde$(nd_end).npy", err_mzmd)
npzwrite("$(dir_path)/cyl_ax0_max_err_mzmd_delay_k1dmd_dts$(t_skip)_st$(sample)_$(method)_$(a_method)_re$(re_num)_modes_nr$(nr)_nk$(nk)_tp$(t_pred_length)_te$(t_end)_nke$(nk_end)_nde$(nd_end).npy", err_mzmd_max)
