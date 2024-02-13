# using Measures

function circle_shape(h,k,r)
    θ=LinRange(0,2*pi,500);
    return h .+ r*sin.(θ), k .+ r*cos.(θ)
end

function plot_field(X, t)
    gsy = 600; gsx = round(Int, 2.25*gsy)
    gr(size=(gsx,gsy))
    t_dim = size(X, 2);
    Y = copy(X);
    Y[abs.(Y).>5] .= 5
    F = reshape(Y, (199, 449, t_dim));
    x_range = range(-1, 8, length=449); y_range = range(-2, 2, length=199);

    plt = plot(x_range, y_range, F[:,:, t]', st=:heatmap,
    # plt = plot(x_range, y_range, F[:,:, t], st=:heatmap,
                bottom_margin=6mm, left_margin=6mm, right_margin=6mm,
                top_margin=6mm, fill=(true, cgrad(:balance)))

    plot!(circle_shape(0,0,.5), seriestype=[:shape], lw = 0.5,
                c = :black, legend=false, fillalpha=0.6, aspectratio=1, clims=(-4.1, 4.1))
    xlims!(-1.06, 8.06); ylims!(-2.06, 2.06);

    title!(L"\textbf{Vorticity}", titlefont=22)
    xlabel!(L"\textbf{x}", xtickfontsize=12, xguidefontsize=20)
    ylabel!(L"\textbf{y}", ytickfontsize=12, yguidefontsize=20)
    display(plt)
    # return plt
end

function plot_field_comp(Xgt, Xpr, t)
    gsy = 1250; gsx = round(Int, 1.2*gsy)
    gr(size=(gsx,gsy))
    t_dim_pr = size(Xpr, 2);
    t_dim_gt = size(Xgt, 2);
    Ygt = copy(Xgt);
    Ygt[abs.(Ygt).>4] .= 4
    Ypr = copy(Xpr);
    Ypr[abs.(Ypr).>4] .= 4
    Fgt = reshape(Ygt, (199, 449, t_dim_gt));
    Fpr = reshape(Ypr, (199, 449, t_dim_pr));

    x_range = range(-1, 8, length=449); y_range = range(-2, 2, length=199);

    plt1 = plot(x_range, y_range, Fgt[:,:, t]', st=:heatmap,
                bottom_margin=6mm, left_margin=6mm, right_margin=6mm,
                top_margin=6mm, fill=(true, cgrad(:balance)))

    plot!(circle_shape(0,0,.5), seriestype=[:shape], lw = 0.5,
                c = :black, legend=false, fillalpha=0.6, aspectratio=1, clims=(-4.1, 4.1))
    xlims!(-1.06, 8.06); ylims!(-2.06, 2.06);

    title!(L"\textbf{Vorticity}", titlefont=22)
    xlabel!(L"\textbf{x}", xtickfontsize=12, xguidefontsize=20)
    ylabel!(L"\textbf{y}", ytickfontsize=12, yguidefontsize=20)

    plt2 = plot(x_range, y_range, Fpr[:,:, t]', st=:heatmap,
    bottom_margin=6mm, left_margin=6mm, right_margin=6mm,
    top_margin=6mm, fill=(true, cgrad(:balance)))

    plot!(circle_shape(0,0,.5), seriestype=[:shape], lw = 0.5,
        c = :black, legend=false, fillalpha=0.6, aspectratio=1, clims=(-4.1, 4.1))
    xlims!(-1.06, 8.06); ylims!(-2.06, 2.06);

    title!(L"\textbf{Vorticity ~ Prediction}", titlefont=22)
    xlabel!(L"\textbf{x}", xtickfontsize=12, xguidefontsize=20)
    ylabel!(L"\textbf{y}", ytickfontsize=12, yguidefontsize=20)

    plt = plot(plt1, plt2, layout=(2,1))
    display(plt)
end


function plot_field_diff_mse(Xgt, Xpr, t, c_max)
    gsy = 1250; gsx = round(Int, 1.2*gsy)
    gr(size=(gsx,gsy))
    t_dim = size(Xpr, 2);
    Ygt = copy(Xgt);
    Ypr = copy(Xpr);
    Fgt = reshape(Ygt, (199, 449));
    Fpr = reshape(Ypr, (199, 449));

    x_range = range(-1, 8, length=449); y_range = range(-2, 2, length=199);

    plt1 = plot(x_range, y_range, Fgt[:,:]', st=:heatmap,
                bottom_margin=6mm, left_margin=6mm, right_margin=6mm,
                top_margin=6mm, fill=(true, cgrad(:balance)), clims=(0, c_max))

    plot!(circle_shape(0,0,.5), seriestype=[:shape], lw = 0.5,
                c = :black, legend=false, fillalpha=0.6, aspectratio=1)
    xlims!(-1.06, 8.06); ylims!(-2.06, 2.06);

    title!(L"\textbf{Vorticity ~ MZMD ~ Pointwise ~ MSE}", titlefont=22)
    xlabel!(L"\textbf{x}", xtickfontsize=12, xguidefontsize=20)
    ylabel!(L"\textbf{y}", ytickfontsize=12, yguidefontsize=20)

    plt2 = plot(x_range, y_range, Fpr[:,:]', st=:heatmap,
    bottom_margin=6mm, left_margin=6mm, right_margin=6mm,
    top_margin=6mm, fill=(true, cgrad(:balance)), clims=(0, c_max))

    plot!(circle_shape(0,0,.5), seriestype=[:shape], lw = 0.5,
        c = :black, legend=false, fillalpha=0.6, aspectratio=1)
    xlims!(-1.06, 8.06); ylims!(-2.06, 2.06);

    title!(L"\textbf{Vorticity ~ DMD ~ Pointwise ~ MSE}", titlefont=22)
    xlabel!(L"\textbf{x}", xtickfontsize=12, xguidefontsize=20)
    ylabel!(L"\textbf{y}", ytickfontsize=12, yguidefontsize=20)

    plt = plot(plt1, plt2, layout=(2,1))
    display(plt)
    # return plt
end

function plot_field_diff(Xgt, Xpr, t)
    gsy = 1250; gsx = round(Int, 1.2*gsy)
    gr(size=(gsx,gsy))
    t_dim_gt = size(Xgt, 2);
    t_dim_pr = size(Xpr, 2);
    Ygt = copy(Xgt);
    Ygt[abs.(Ygt).>4] .= 4
    Ypr = copy(Xpr);
    Ypr[abs.(Ypr).>4] .= 4
    Fgt = reshape(Ygt, (199, 449, t_dim_gt));
    Fpr = reshape(Ypr, (199, 449, t_dim_pr));

    x_range = range(-1, 8, length=449); y_range = range(-2, 2, length=199);

    plt1 = plot(x_range, y_range, Fgt[:,:, t]', st=:heatmap,
                bottom_margin=6mm, left_margin=6mm, right_margin=6mm,
                top_margin=6mm, fill=(true, cgrad(:balance)))

    plot!(circle_shape(0,0,.5), seriestype=[:shape], lw = 0.5,
                c = :black, legend=false, fillalpha=0.6, aspectratio=1)
    xlims!(-1.06, 8.06); ylims!(-2.06, 2.06);

    title!(L"\textbf{Vorticity ~ MZMD ~ Difference}", titlefont=22)
    xlabel!(L"\textbf{x}", xtickfontsize=12, xguidefontsize=20)
    ylabel!(L"\textbf{y}", ytickfontsize=12, yguidefontsize=20)

    plt2 = plot(x_range, y_range, Fpr[:,:, t]', st=:heatmap,
    bottom_margin=6mm, left_margin=6mm, right_margin=6mm,
    top_margin=6mm, fill=(true, cgrad(:balance)))

    plot!(circle_shape(0,0,.5), seriestype=[:shape], lw = 0.5,
        c = :black, legend=false, fillalpha=0.6, aspectratio=1)
    xlims!(-1.06, 8.06); ylims!(-2.06, 2.06);

    title!(L"\textbf{Vorticity ~ DMD ~ Difference}", titlefont=22)
    xlabel!(L"\textbf{x}", xtickfontsize=12, xguidefontsize=20)
    ylabel!(L"\textbf{y}", ytickfontsize=12, yguidefontsize=20)

    plt = plot(plt1, plt2, layout=(2,1))
    display(plt)
    # return plt
end


function plotting_sing_val(r_max, S)
    gr(size=(700,600))
    rs = 1 : r_max
    plt = scatter(rs, S[1:r_max], yaxis =:log, label=false,
        ms=7.5, legendfontsize=10, color="black")
    title!(L"\textrm{Singular Values}", titlefont=22)
    xlabel!(L"\textrm{r}", xtickfontsize=12, xguidefontsize=16)
    ylabel!(L"\sigma_k", ytickfontsize=12, yguidefontsize=16)
    display(plt)
    # savefig(plt, "./figures/singular_values_r.png")
end

function plot_mzmd_Mmodes(Φ, r, m)
    gsy = 600; gsx = round(Int, 2.25*gsy)
    gr(size=(gsx,gsy));
    nx = 199; ny = 449;
    Φ = reshape(Φ, (199, 449, r));
    x_range = range(-1, 8, length=449); y_range = range(-2, 2, length=199);

    plt = plot(x_range, y_range, real.(Φ[:,:, m])', st=:heatmap,
                bottom_margin=6mm, left_margin=6mm, right_margin=6mm,
                top_margin=6mm, fill=(true, cgrad(:balance)))

    plot!(circle_shape(0,0,.5), seriestype=[:shape], lw = 0.5,
                c = :black, legend=false, fillalpha=0.6, aspectratio=1)
    xlims!(-1.06, 8.06); ylims!(-2.06, 2.06);

    mode_n = ceil(Int, m/2)
    title!(L"\textrm{Markovian ~ (DMD) ~ mode ~ } \Phi^{m}_{%$mode_n}", titlefont=22)
    xlabel!(L"\textbf{x}", xtickfontsize=12, xguidefontsize=20)
    ylabel!(L"\textbf{y}", ytickfontsize=12, yguidefontsize=20)
    # savefig(plt, "./figures/mzmd_Markovian_mode$(m)_nk$(n_ks)_r$(r).png")
    return plt
end

function plot_mzmd_modes(Φ, r, n_ks, m)
    gsy = 600; gsx = round(Int, 2.25*gsy)
    gr(size=(gsx,gsy));
    nx = 199; ny = 449;
    Φ = reshape(Φ, (199, 449, n_ks*r));
    r_2 = floor(Int, n_ks*r/2)

    x_range = range(-1, 8, length=449); y_range = range(-2, 2, length=199);

    plt = plot(x_range, y_range, real.(Φ[:,:, m])', st=:heatmap,
                bottom_margin=6mm, left_margin=6mm, right_margin=6mm,
                top_margin=6mm, fill=(true, cgrad(:balance)))

    plot!(circle_shape(0,0,.5), seriestype=[:shape], lw = 0.5,
                c = :black, legend=false, fillalpha=0.6, aspectratio=1)
    xlims!(-1.06, 8.06); ylims!(-2.06, 2.06);

    mode_n = ceil(Int, m/2)
    title!(L"\textrm{MZMD ~ mode ~ } \Phi^{mz}_{%$mode_n}", titlefont=22)
    xlabel!(L"\textbf{x}", xtickfontsize=12, xguidefontsize=20)
    ylabel!(L"\textbf{y}", ytickfontsize=12, yguidefontsize=20)
    # savefig(plt, "./figures/mzmd_Markovian_mode$(m)_nk$(n_ks)_r$(r).png")
    return plt
end

function plot_amplitude_vs_frequency_markovian(λ, b, r, k)
    gr(size=(700,600));
    # r_2 = floor(Int, r/2)
    r_2 = r
    # plt = scatter(imag(λ)[r_2:end], abs.(b)[r_2:end], linewidth=2.2, legend=false, ms=6.5)
    plt = scatter(imag(λ)[1:r_2], abs.(b)[1:r_2], linewidth=2.2, legend=false, ms=6.5)
    title!(L"\textrm{ Amplitude ~ vs ~ Frequency ~ } \mathbf{\Omega}^{(0)}", titlefont=18)
    xlabel!(L"\textrm{Im}(\lambda)", xtickfontsize=14, xguidefontsize=14)
    ylabel!(L"a_n", ytickfontsize=14, yguidefontsize=14)
    # savefig(plt, "./figures/markovian_cylinder_spectrum_r$(r)_k$(k).png")
    return plt
end


function plot_amplitude_vs_frequency_mzmd(λ, a, r, n_ks)
    gr(size=(700,600));
    r_2 = floor(Int, n_ks*r/2)
    # plt = scatter(imag(λ)[r_2:end], abs.(a)[r_2:end], legend=false, ms=6.5)
    plt = scatter(imag(λ)[1:r_2], abs.(a)[1:r_2], legend=false, ms=6.5)

    title!(L"\textrm{ Amplitude ~ vs ~ Frequency ~ } C", titlefont=18)
    xlabel!(L"\textrm{Im}(\lambda)", xtickfontsize=14, xguidefontsize=14)
    ylabel!(L"a_n", ytickfontsize=14, yguidefontsize=14)
    return plt
end

function plotting_mem_norm(Ω, M, n_ks)
    gr(size=(700,600))
    Ω_norms = zeros(n_ks)
    for i in 1:n_ks
        Ω_norms[i] = norm(Ω[i,:,:])
    end
    y_vals = Ω_norms[1:end] / norm(M);
    plt_omega = Plots.scatter(y_vals, yaxis=:log, label=false, ms=7.5, color="black")
    title!(L"\textrm{Memory ~ Effects:   } k = %$n_ks", titlefont=22)
    xlabel!(L"k", xtickfontsize=12, xguidefontsize=16)
    ylabel!(L"||\Omega^{(k)}||_F / ||\Omega^{(0)}||_F", ytickfontsize=12, yguidefontsize=16)
    display(plt_omega)
end

function plot_c_spectrum(Λc)
    gr(size=(600,600));
    function circle_shape(h,k,r)
        θ=LinRange(0,2*pi,500);
        return h .+ r*sin.(θ), k .+ r*cos.(θ)
    end
    plt = scatter(Λc, ms = 4.0, color = "black", legend=false)
    plot!(circle_shape(0,0,1.0), linewidth=1.5, color="black", linestyle=:dash)
    title!(L"\mathbf{C} \textrm{ ~ Eigenvalues }", titlefont=22)
    xlabel!(L"\textrm{Re}(\lambda)", xtickfontsize=12, xguidefontsize=16)
    ylabel!(L"\textrm{Im}(\lambda)", ytickfontsize=12, yguidefontsize=16)
    display(plt)
end
