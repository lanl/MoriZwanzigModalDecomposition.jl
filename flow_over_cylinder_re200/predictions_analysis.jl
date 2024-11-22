using Plots, LaTeXStrings, LinearAlgebra, NPZ
using Statistics


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

# gx = load_grid_file(nx)
m = size(X, 1);
t_end = 120 #total n steps is 751
t_pred = 20
T_train = ((sample):(t_end+sample-1))[1:t_skip:end];
T_test = ((t_end+sample):(t_end + t_pred + sample))[1:t_skip:end];
time_train = dt * T_train
time_test = dt * t_skip * T_test;
t_all = 0:dt:(dt*(t_end + t_pred +1));
X_train = X[:,T_train];
X_test = X[:,T_test];


t_ = 20
t_r = round((dt*(t_-2)), digits=2)

#-------- Set params
#number of memory terms:
n_ks = 18
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

plt = plot(Ω_norms, yaxis=:log)
display(plt)

C = form_companion(Ω, r, n_ks);

#----------------------------------------------------------------
                        #future state predictions
#----------------------------------------------------------------


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


nmodes_ = "all"
# nmodes_ = 1000;
lam, a, Q, Ur = mzmd_modes_reduced_amps_for_errors(t_win, X, r, k)
if nmodes_=="all"
    num_modes = k*r;
else
    num_modes = minimum([k*r, nmodes_]);
end
a_dom, Vc_dom, lam_dom = select_dominant_modes(a, Ur, Q, r, k, lam, num_modes) #dominant three modes
solution_modes = simulate_modes_time_delay(Ur, Vc_dom, a_dom, r, lam_dom, dt, time_test)


