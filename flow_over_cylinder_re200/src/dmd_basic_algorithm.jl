using LinearAlgebra

#see DMD book by Brunton et al. for more details.
#This is Tu "exact" dmd approach

function dmd_alg(X1, X2, r, dt)
    #inputs: X1 data matrix
            #X2 shifted data matrix
            #r target rank of SVD
            #dt time step

    #outputs: Φ DMD modes
             #ω = cont time DMD e.values
             #λ, discrete time dmd e.values
             #b vector of magnitutes of modes Φ

    U, S, V = svd(X1);
    r = minimum([r, size(U)[2]])

    U_r = U[:, 1:r]
    S = Diagonal(S)
    S_r = S[1:r, 1:r];
    V_r = V[:, 1:r];

    Atilde = U_r'*X2*V_r / S_r
    lambda, W_r = eigen(Atilde);
    Phi = X2 * V_r / S_r * W_r;

    omega = log.(lambda)/dt;
    x1 = X1[:, 1];
    b = Phi\x1;

    mm1 = size(X1)[2]; time_dynamics = complex(zeros(r, mm1));
    ts = (0:mm1-1) * dt;
    for i in 1 : mm1
        time_dynamics[:,i] = (b.*exp.(ts[i] .* omega));
    end
    Xdmd = Phi * time_dynamics;
    return Xdmd, Phi, omega, lambda, b
end

function extract_dmd_modes(X1, X2, r, dt)
    U, S, V = svd(X1);
    r = minimum([r, size(U)[2]])

    U_r = U[:, 1:r]
    S = Diagonal(S)
    S_r = S[1:r, 1:r];
    V_r = V[:, 1:r];

    Atilde = U_r'*X2*V_r / S_r
    lambda, W_r = eigen(Atilde);
    Phi = X2 * V_r / S_r * W_r;
    omega = log.(lambda)/dt;

    return Phi, omega, lambda
end

function dmd_prediction(Phi, omega, x1, t, dt, r)
    time_dynamics = complex(zeros(r, t));
    ts = (0:t-1) * dt; b = Phi\x1;
    for i in 1 : t
        time_dynamics[:,i] = (b.*exp.(ts[i] .* omega));
    end
    Xdmd = Phi * time_dynamics;
    return Xdmd
end


function dmd_prediction_given_X(X1, X2, x1, t, dt, r)
    U, S, V = svd(X1);
    r = minimum([r, size(U)[2]])

    U_r = U[:, 1:r]
    S = Diagonal(S)
    S_r = S[1:r, 1:r];
    V_r = V[:, 1:r];

    Atilde = U_r'*X2*V_r / S_r
    lambda, W_r = eigen(Atilde);
    Phi = X2 * V_r / S_r * W_r;
    omega = log.(lambda)/dt;

    time_dynamics = complex(zeros(r, t));
    ts = (0:t-1) * dt; b = Phi\x1;
    for i in 1 : t
        time_dynamics[:,i] = (b.*exp.(ts[i] .* omega));
    end
    Xdmd = Phi * time_dynamics;
    return Xdmd
end
