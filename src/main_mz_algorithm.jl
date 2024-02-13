using Plots, LaTeXStrings, LinearAlgebra, NPZ
using Statistics, Measures

function svd_low_rank_proj_observables(X1, X, r; subtractmean::Bool = false)
    if subtractmean X .-= mean(X,dims=2); end
    U, S, V = svd(X1);
    r = minimum([r, size(U)[2]]);

    U_r = U[:, 1:r];
    S = Diagonal(S);
    S_r = S[1:r, 1:r];
    V_r = V[:, 1:r];
    X_proj = U_r' * X;
    return U_r, X_proj
end

function svd_method_of_snapshots(X, r; subtractmean::Bool = false)
    if subtractmean X .-= mean(X,dims=2); end
    #number of snapshots
    m = size(X, 2);
    #Correlation matrix
    C = X'*X;

    #Eigen decomp
    E = eigen!(C);
    e_vals = E.values;
    e_vecs = E.vectors;
    #sort eigenvectors
    sort_ind = sortperm(abs.(e_vals)/m, rev=true)
    e_vecs = e_vecs[:,sort_ind];
    e_vals = e_vals[sort_ind];
    #singular values
    S = sqrt.(abs.(e_vals));
    #modes and coefficients
    U_r = X*e_vecs*Diagonal(1 ./S)[:, 1:r]
    # a = Diagonal(S)*e_vecs'
    return S, U_r, U_r' * X
end




function obtain_C(X, t_win, n_ks)
    ty = typeof(X[1,1]); m = size(X)[1];
    C = zeros(ty, n_ks+1, m, m);
    for δ in 0 : n_ks
        C[δ+1,:,:] = X[:, 1+δ : t_win+δ] * X[:, 1 : t_win]';
    end
    return C
end


function sum_term_C(X, t_win, i, j, k)
    ty = typeof(X[1,1]);
    S = zeros(ty, 1)[1];
    for l in 1 : (t_win-k)
        S += X[i, k+l] * X[j, l]
    end
    return 1/(t_win - k) * S
end


function sum_term_ker(Ω, C, k)
    ty = typeof(Ω[1,1,1]); m, m2 = size(Ω[1,:,:]);
    S = zeros(ty, m, m2);
    for l in 1 : k - 1
        S += Ω[l,:,:] * C[k - l + 1, :, :];
    end
    return S
end

function obtain_ker(C, n_ks)
    ty = typeof(C[1,1,1]);
    Ω = zeros(ty, size(C)[1]-1, size(C)[2], size(C)[3]);
    C0_inv = pinv(C[1,:,:]);
    Ω[1,:,:] = C[2,:,:] * C0_inv;
    M = Ω[1,:,:];
    if n_ks > 1
        for k in 2 : n_ks
            S = sum_term_ker(Ω, C, k);
            Ω[k, :, :] = (C[k+1,:,:] - S) * C0_inv;
        end
    end
    return M, Ω
end




"""
    Compute predictions
    g((k+1)Δt) = Σ Ω(l) ⋅ g((k-l)Δt) + Wₖ

    For now assume Wₖ is zero
"""

function sum_term_pred(X, Ω, k, n_ks)
    ty = typeof(X[1,1]);
    S = zeros(ty, size(X[:,1])[1]);
    for l in 1 : minimum([k, n_ks])
        S += Ω[l,:,:] * X[:, k - l + 1];
    end
    return S
end


function obtain_prediction(X, Ω, n_ks)
    ty = typeof(X[1,1]);
    X_pred = zeros(ty, size(X)[1], size(X)[2])
    X_pred[:, 1] = X[:, 1];
    for k in 1 : size(X)[2] - 1
        X_pred[:, k+1] = sum_term_pred(X, Ω, k, n_ks) #+ W;
    end
    return X_pred
end


function obtain_future_state_prediction(Ur, X1_gen, Ω, n_ks, T_pred)
    ty = typeof(X1_gen[1,1]);
    X_pred = zeros(ty, size(X1_gen)[1], T_pred+n_ks)
    #initial condition including history:
    X_pred[:, 1:(n_ks)] = X1_gen[:, 1:(n_ks)];
    for k in (n_ks) : (T_pred + n_ks - 1)
        X_pred[:, k+1] = sum_term_pred(X_pred, Ω, k, n_ks)
    end
    return X_pred
end


function obtain_norm_mem(Ω, n_ks)
    out = zeros(n_ks)
    for i in 1 : n_ks
        out[i] = norm(Ω[i,:,:])
    end
    return out
end
