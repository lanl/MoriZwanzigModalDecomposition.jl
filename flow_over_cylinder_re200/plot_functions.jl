using Measures

function circle_shape(h,k,r)
    θ=LinRange(0,2*pi,500);
    return h .+ r*sin.(θ), k .+ r*cos.(θ)
end

function plot_field(X, t, method)
    gsy = 600; gsx = round(Int, 2.25*gsy)
    gr(size=(gsx,gsy), dpi=300)
    t_dim = size(X, 2);
    Y = copy(X);
    Y[abs.(Y).>5] .= 5
    F = reshape(Y, (199, 449, t_dim));
    x_range = range(-1, 8, length=449); y_range = range(-2, 2, length=199);

    plt = plot(x_range, y_range, F[:,:, t]', st=:heatmap,
    # plt = plot(x_range, y_range, F[:,:, t], st=:heatmap,
                bottom_margin=6mm, left_margin=6mm, right_margin=6mm,
                top_margin=6mm, fill=(true, cgrad(:balance)))
    scatter!([x_range[200]], [y_range[100]], color="black", ms=4)
    plot!(circle_shape(0,0,.5), seriestype=[:shape], lw = 0.5,
                c = :black, legend=false, fillalpha=0.6, aspectratio=1, clims=(-8.1, 8.1))
    xlims!(-1.06, 8.06); ylims!(-2.06, 2.06);

    #title!(L"\textbf{Vorticity}", titlefont=22)
    title!(L"\textrm{%$(method)}", titlefont=22)
    xlabel!(L"\textbf{x}", xtickfontsize=12, xguidefontsize=20)
    ylabel!(L"\textbf{y}", ytickfontsize=12, yguidefontsize=20)
    display(plt)
    return plt
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

function plot_field_diff_mse_mz_hodmd_dmd(Xgt, Xpr, Xdmd, t, c_max)
    gsy = 1250; gsx = round(Int, 1.2*gsy)
    gr(size=(gsx,gsy))
    t_dim = size(Xpr, 2);
    Ygt = copy(Xgt);
    Ypr = copy(Xpr);
    Ydmd = copy(Xdmd);
    Fgt = reshape(Ygt, (199, 449));
    Fpr = reshape(Ypr, (199, 449));
    Fdmd = reshape(Ydmd, (199, 449));

    x_range = range(-1, 8, length=449); y_range = range(-2, 2, length=199);

    plt1 = plot(x_range, y_range, Fgt[:,:]', st=:heatmap,
                bottom_margin=6mm, left_margin=6mm, right_margin=6mm,
                top_margin=6mm, fill=(true, cgrad(:balance)), clims=(0, c_max))

    plot!(circle_shape(0,0,.5), seriestype=[:shape], lw = 0.5,
                c = :black, legend=false, fillalpha=0.6, aspectratio=1)
    xlims!(-1.06, 8.06); ylims!(-2.06, 2.06);

    title!(L"\textbf{Vorticity ~ MZMD ~ MSE}", titlefont=22)
    xlabel!(L"\textbf{x}", xtickfontsize=12, xguidefontsize=20)
    ylabel!(L"\textbf{y}", ytickfontsize=12, yguidefontsize=20)

    plt2 = plot(x_range, y_range, Fpr[:,:]', st=:heatmap,
    bottom_margin=6mm, left_margin=6mm, right_margin=6mm,
    top_margin=6mm, fill=(true, cgrad(:balance)), clims=(0, c_max))

    plot!(circle_shape(0,0,.5), seriestype=[:shape], lw = 0.5,
        c = :black, legend=false, fillalpha=0.6, aspectratio=1)
    xlims!(-1.06, 8.06); ylims!(-2.06, 2.06);

    title!(L"\textbf{Vorticity ~ HODMD ~ MSE}", titlefont=22)
    xlabel!(L"\textbf{x}", xtickfontsize=12, xguidefontsize=20)
    ylabel!(L"\textbf{y}", ytickfontsize=12, yguidefontsize=20)

    #DMD
    plt3 = plot(x_range, y_range, Fdmd[:,:]', st=:heatmap,
    bottom_margin=6mm, left_margin=6mm, right_margin=6mm,
    top_margin=6mm, fill=(true, cgrad(:balance)), clims=(0, c_max))

    plot!(circle_shape(0,0,.5), seriestype=[:shape], lw = 0.5,
        c = :black, legend=false, fillalpha=0.6, aspectratio=1)
    xlims!(-1.06, 8.06); ylims!(-2.06, 2.06);

    title!(L"\textbf{Vorticity ~ DMD ~ MSE}", titlefont=22)
    xlabel!(L"\textbf{x}", xtickfontsize=12, xguidefontsize=20)
    ylabel!(L"\textbf{y}", ytickfontsize=12, yguidefontsize=20)

    plt = plot(plt1, plt2, plt3,  layout=(3,1))
    display(plt)
    # return plt
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

    title!(L"\textbf{Vorticity ~ MZMD ~ MSE}", titlefont=22)
    xlabel!(L"\textbf{x}", xtickfontsize=12, xguidefontsize=20)
    ylabel!(L"\textbf{y}", ytickfontsize=12, yguidefontsize=20)

    plt2 = plot(x_range, y_range, Fpr[:,:]', st=:heatmap,
    bottom_margin=6mm, left_margin=6mm, right_margin=6mm,
    top_margin=6mm, fill=(true, cgrad(:balance)), clims=(0, c_max))

    plot!(circle_shape(0,0,.5), seriestype=[:shape], lw = 0.5,
        c = :black, legend=false, fillalpha=0.6, aspectratio=1)
    xlims!(-1.06, 8.06); ylims!(-2.06, 2.06);

    title!(L"\textbf{Vorticity ~ DMD ~ MSE}", titlefont=22)
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

function plot_mzmd_Mmodes(Φ, m)
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

function plot_hodmd_modes(Φ, m)
    gsy = 600; gsx = round(Int, 2.25*gsy)
    gr(size=(gsx,gsy));
    nx = 199; ny = 449;
    Φ = reshape(Φ, (nx, ny, r));
    r_2 = floor(Int, n_ks*r/2)

    x_range = range(-1, 8, length=ny); y_range = range(-2, 2, length=nx);

    plt = plot(x_range, y_range, real.(Φ[:,:, m])', st=:heatmap,
                bottom_margin=6mm, left_margin=6mm, right_margin=6mm,
                top_margin=6mm, fill=(true, cgrad(:balance)))

    plot!(circle_shape(0,0,.5), seriestype=[:shape], lw = 0.5,
                c = :black, legend=false, fillalpha=0.6, aspectratio=1)
    xlims!(-1.06, 8.06); ylims!(-2.06, 2.06);

    mode_n = ceil(Int, m/2)
    title!(L"\textrm{HODMD ~ mode ~ } \Phi^{mz}_{%$mode_n}", titlefont=22)
    xlabel!(L"\textbf{x}", xtickfontsize=12, xguidefontsize=20)
    ylabel!(L"\textbf{y}", ytickfontsize=12, yguidefontsize=20)
    # savefig(plt, "./figures/mzmd_Markovian_mode$(m)_nk$(n_ks)_r$(r).png")
    return plt
end


function plot_energy_content(s,r,t_win, eps=0.99)
    gr(size=(700,600));
    Λ = 1/t_win * s.^2;
    energy_r = zeros(r);
    for i in 1 : r
        energy_r[i] = sum(Λ[1:i])/sum(Λ);
    end
    plt = scatter(energy_r, ms=5.0, label=false, ylims=(0.0, 1.1),
                legendfontsize=12, color="black")
    y_99 = eps * ones(r);
    per_eps = 100*eps;
    plot!(y_99, linewidth=3.0, label=L"%$per_eps \textbf{\% ~ variations}", ls=:dash, color="grey")
    title!(L"\textbf{Total ~ variation ~ contained ~ in ~ POD ~ modes}", titlefont=20)
    xlabel!(L"\textbf{r}", xtickfontsize=12, xguidefontsize=16)
    ylabel!(L"\textbf{Total variation}", ytickfontsize=12, yguidefontsize=16)
    # savefig(plt, "./figures/pod_mode_energy_ts$(t_skip)_nt$(nt_samp)_r$(r).png")
    display(plt)
    # return plt
end



function plot_mzmd_proj_memory_modes(Φ, m)
    gsy = 600; gsx = round(Int, 2.25*gsy)
    gr(size=(gsx,gsy));
    nx = 199; ny = 449;
    Φ = reshape(Φ, (199, 449, r));
    r_2 = floor(Int, n_ks*r/2)

    x_range = range(-1, 8, length=449); y_range = range(-2, 2, length=199);

    plt = plot(x_range, y_range, real.(Φ[:,:, m])', st=:heatmap,
                bottom_margin=6mm, left_margin=6mm, right_margin=6mm,
                top_margin=6mm, fill=(true, cgrad(:balance)))

    plot!(circle_shape(0,0,.5), seriestype=[:shape], lw = 0.5,
                c = :black, legend=false, fillalpha=0.6, aspectratio=1)
    xlims!(-1.06, 8.06); ylims!(-2.06, 2.06);

    mode_n = ceil(Int, m/2)
    title!(L"\textrm{MZMD ~ Memory ~ modes ~ } \Phi^{mz}_{%$mode_n}", titlefont=22)
    xlabel!(L"\textbf{x}", xtickfontsize=12, xguidefontsize=20)
    ylabel!(L"\textbf{y}", ytickfontsize=12, yguidefontsize=20)
    # savefig(plt, "./figures/mzmd_Markovian_mode$(m)_nk$(n_ks)_r$(r).png")
    return plt
end

function plot_mzmd_memory_modes(Φ, m)
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
    title!(L"\textrm{MZMD ~ Memory ~ modes ~ } \Phi^{mz}_{%$mode_n}", titlefont=22)
    xlabel!(L"\textbf{x}", xtickfontsize=12, xguidefontsize=20)
    ylabel!(L"\textbf{y}", ytickfontsize=12, yguidefontsize=20)
    # savefig(plt, "./figures/mzmd_Markovian_mode$(m)_nk$(n_ks)_r$(r).png")
    return plt
end

function plot_mzmd_modes(Φ, m)
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

function plot_amplitude_vs_frequency_markovian(λ, b, k)
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


function plot_amplitude_vs_frequency_mzmd(λ, a)
    gr(size=(700,600));
    kr = size(λ,1);
    r_2 = floor(Int, kr/2)
    # plt = scatter(imag(λ)[r_2:end], abs.(a)[r_2:end], legend=false, ms=6.5)
    # plt = scatter(imag(λ)[1:r_2], abs.(a)[1:r_2], legend=false, ms=6.5)
    plt = scatter(imag(λ), abs.(a), legend=false, ms=6.5)

    title!(L"\textrm{ Amplitude ~ vs ~ Frequency ~ } C", titlefont=18)
    xlabel!(L"\textrm{Im}(\lambda)", xtickfontsize=14, xguidefontsize=14)
    ylabel!(L"a_n", ytickfontsize=14, yguidefontsize=14)
    # savefig(plt, "./figures/markovian_cylinder_spectrum_r$(r)_k$(k).png")
    return plt
end

function plotting_mem_norm(Ω)
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
    # savefig(plt_omega, "./figures/mem_mz_norm_nd$(n_ks)_r$(r).png")
end

function plot_c_spectrum(Λc)
    gr(size=(570,570), xtickfontsize=12, ytickfontsize=12, xguidefontsize=20, yguidefontsize=14, legendfontsize=12,
        dpi=300, grid=(:y, :gray, :solid, 1, 0.4), palette=cgrad(:plasma, 3, categorical = true));

    # Λc, Vc = load_py_eigen()
    function circle_shape(h,k,r)
        θ=LinRange(0,2*pi,500);
        return h .+ r*sin.(θ), k .+ r*cos.(θ)
    end
    plt = scatter(Λc, linewidth=2.0, ms = 4.0, color = "black", legend=false, grid = true, framestyle = :box)
    plot!(circle_shape(0,0,1.0), linewidth=1.5, color="black", linestyle=:dash)
    title!(L"\textrm{MZMD ~ Eigenvalues }", titlefont=22)
    xlabel!(L"\textrm{Re}(\lambda)", xtickfontsize=12, xguidefontsize=16)
    ylabel!(L"\textrm{Im}(\lambda)", ytickfontsize=12, yguidefontsize=16)
    return plt
end

function plot_dmd_spectrum(Λc)
    gr(size=(570,570), xtickfontsize=12, ytickfontsize=12, xguidefontsize=20, yguidefontsize=14, legendfontsize=12,
        dpi=300, grid=(:y, :gray, :solid, 1, 0.4), palette=cgrad(:plasma, 3, categorical = true));

    # Λc, Vc = load_py_eigen()
    function circle_shape(h,k,r)
        θ=LinRange(0,2*pi,500);
        return h .+ r*sin.(θ), k .+ r*cos.(θ)
    end
    plt = scatter(Λc, linewidth=2.0, ms = 4.0, color = "black", legend=false, grid = true, framestyle = :box)
    plot!(circle_shape(0,0,1.0), linewidth=1.5, color="black", linestyle=:dash)
    title!(L"\textrm{DMD ~ Eigenvalues }", titlefont=22)
    xlabel!(L"\textrm{Re}(\lambda)", xtickfontsize=12, xguidefontsize=16)
    ylabel!(L"\textrm{Im}(\lambda)", ytickfontsize=12, yguidefontsize=16)
    return plt
end


function plot_energy_content(s,r)
    gr(size=(570,300), xtickfontsize=12, ytickfontsize=12, xguidefontsize=20, yguidefontsize=14, legendfontsize=12,
        dpi=300, grid=(:y, :gray, :solid, 1, 0.4), palette=cgrad(:plasma, 3, categorical = true));

    Λ = 1/t_win * s.^2;
    energy_r = zeros(r);
    for i in 1 : r
        energy_r[i] = sum(Λ[1:i])/sum(Λ);
    end
    plt = plot(energy_r, lw=3.0, label=false, ylims=(0.0, 1.0),
                legendfontsize=12, color="black")
    y_99 = 0.99 * ones(r);
    plot!(y_99, linewidth=3.0, label=L"\textbf{99\% ~ variations}", ls=:dash, color="grey")
    title!(L"\textrm{Total ~ variation ~ in ~ POD ~ modes}", titlefont=22)
    xlabel!(L"\textrm{r}", xtickfontsize=12, xguidefontsize=16)
    ylabel!(L"\textrm{Voriticy ~ variation}", ytickfontsize=12, yguidefontsize=16)
    savefig(plt, "./figures/pod_mode_energy_ts$(t_skip)_r$(r)_k$(n_ks).png")
    return plt
end