

# construct time delay obserables in projected (x_proj) space. Use this to find operators and e.vects
function compute_modes_amplitudes_time_delay_observables(X, T_train, r, n_ks, n_td, T, method, a_method)
    #inputs: r number of observables (pod modes)
            #n_ks number of mz memory terms
            #n_td number of time delay embeddings
    S, Ur, X_proj = svd_method_of_snapshots(X[:,T_train], r, subtractmean=true)
    # %% Create time delay observable matrix
    t_g = T + n_ks + 1;
    t_gtilde = length(1:(1+t_g - n_td));
    G_tilde=zeros(n_td*r, t_gtilde);
    for i in 1:n_td
        G_tilde[(i-1)*r+1:i*r,:] = X_proj[:,i:(i+t_g-n_td)];
    end
    #initial conditions for z0 in companion 
    # g0_test = Ur'*X_test[:, 1:(n_ks+n_td+1)];
    # G0_tilde_test = zeros(n_td*r, n_ks);
    # for i in 1:n_td
    #     G0_tilde_test[(i-1)*r+1:i*r,:] = g0_test[:,i:(i+n_ks-1)];
    # end

    function obtain_mz_operators_(G, t_win, n_ks)
        #compute two time covariance matrices
        Cov = obtain_C(G, t_win, n_ks);
        #compute MZ operators with Mori projection
        _, Ω = obtain_ker(Cov, n_ks);
        println("----------------------------");
        return Ω
    end
    twin_tilde = t_gtilde - n_ks - 1;
    if n_ks==1 #hodmd (i.e. no mz memory)
        # #TODO SVD 2
        U1, Sigma1, T1 = svd(G_tilde); 
        sigmas1=diagm(Sigma1); 
        # Second Spatial dimension reduction
        r2 = n_td*r
        if r2 >= T_train[end]
            r2 = size(U1, 2); #this will break companion structure, but is a limit of this method!
        end
        U1=U1[:,1:r2];
        #svd low rank approximation of G_tilde:
        hatG=sigmas1[1:r2,1:r2]*T1[:,1:r2]';
        # #TODO SVD 3
        K1 = size(hatG, 2);
        tildeU1, tildeSigma, tildeV1 = svd(hatG[:,1:K1-1]);
        tildesigmas = diagm(tildeSigma)
        Ω_tilde=hatG[:,2:K1]*tildeV1*inv(tildesigmas)*tildeU1';
        lam, e_vects = eigen(Ω_tilde);
        Q=U1*e_vects;
        if a_method=="lsa"
            a, Q = compute_amplitudes_observability_matrix_aq(Q, lam, G_tilde);
        else
            a, Q = compute_amplitudes_given_ic_z0_hodmd(Ur, X_test[:, 1:n_td], Q, n_td, r)
        end
    else 
        Ω_tilde = obtain_mz_operators_(G_tilde, twin_tilde, n_ks);
        if method=="nep"
            lam, e_vects = compute_nonlinear_eigen(Ω_tilde); 
        else
            r2 = size(Ω_tilde, 2);
            C = form_companion(Ω_tilde, r2, n_ks);
            lam, Vc = eigen(C);
        end
        if a_method=="lsa"
            e_vects = Vc[1:r2,:];
            a, Q = compute_amplitudes_observability_matrix_aq(e_vects, lam, G_tilde);
        else
            a, Q = compute_amplitudes_given_ic_z0(G0_tilde_test, Vc, n_ks, n_td, r)
        end
    end
    return lam, Q, a, Ur, Ω_tilde
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