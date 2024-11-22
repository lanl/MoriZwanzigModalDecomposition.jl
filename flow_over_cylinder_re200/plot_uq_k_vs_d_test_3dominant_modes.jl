using NPZ
using Plots
using LaTeXStrings
using Measures
using Statistics
using Colors, ColorSchemes

# data_type = "recon"
data_type = "gen"
# error_type = "max"
error_type = "mse"
dir = "dom_modes_gen_error_x0_test_dt2_te100_re600_nm6";

re_num = 200;
dt_skip = 2;
samples = 10:10:220;
nr = 4
nk = 13
tp = 50
te = 100;
nke = 24

#select time range for average errors
t_range = 1:10
t = 1
d = 1
#select this r index
r = 3
ks = 1;
am = "x0"
r_range = 3:2:9;
nk_range = cat(1:2, 4:2:24, dims=1)
nd_range = cat(1:2, 4:2:24, dims=1)



function obtain_all_samples(samps)
    if error_type=="max"
        mzmd_err = npzread("$(dir)/cyl_ax0_max_err_mzmd_delay_k1dmd_dts$(dt_skip)_st$(samps[1])_companion_x0_re$(re_num)_modes_nr$(nr)_nk$(nk)_tp$(tp)_te$(te)_nke$(nke)_nde1.npy");
    else
        mzmd_err = npzread("$(dir)/cyl_ax0_err_mzmd_delay_k1dmd_dts$(dt_skip)_st$(samps[1])_companion_x0_re$(re_num)_modes_nr$(nr)_nk$(nk)_tp$(tp)_te$(te)_nke$(nke)_nde1.npy");
    end
    T, nks, nds, nrs = size(mzmd_err)
    ns = size(samps,1);
    mzmd_errs_samps = zeros(ns, T, nks, nds, nrs)
    mzmd_errs_samps[1,:,:,:,:] = mzmd_err 
    for i in 2:ns
        s = samps[i];
        if error_type=="max"
            mzmd_err = npzread("$(dir)/cyl_ax0_max_err_mzmd_delay_k1dmd_dts$(dt_skip)_st$(s)_companion_x0_re$(re_num)_modes_nr$(nr)_nk$(nk)_tp$(tp)_te$(te)_nke$(nke)_nde1.npy");
        else
            mzmd_err = npzread("$(dir)/cyl_ax0_err_mzmd_delay_k1dmd_dts$(dt_skip)_st$(s)_companion_x0_re$(re_num)_modes_nr$(nr)_nk$(nk)_tp$(tp)_te$(te)_nke$(nke)_nde1.npy");
        end
        mzmd_errs_samps[i,:,:,:,:] = mzmd_err 
    end
    return abs.(mzmd_errs_samps)
end



mzmd_errs_samps = obtain_all_samples(samples)

mzmd_errs_samps = mean(mzmd_errs_samps[:, t_range[1:end], :, :, :], dims=2)[:,1,:,:,:];


ns, nks, nds, nrs = size(mzmd_errs_samps)
println(size(mzmd_errs_samps))
dt = dt_skip*0.2


rs = r_range[r]
ds = nd_range[d]
println("selected r = ", rs)
println("selected d = ", ds)


function plot_k_lines_uq(nk_range, nd_range, errk)
	gr(size=(570,450), xtickfontsize=12, ytickfontsize=12, xguidefontsize=20, yguidefontsize=14, legendfontsize=12,
		dpi=300, grid=(:y, :gray, :solid, 1, 0.4), palette=cgrad(:plasma, 3, categorical = true));
    xs = nk_range.-1;
    err_samp_selected = errk[:,:,1,r];
    μs = mean(err_samp_selected,dims=1)[1,:];
    σs = std(err_samp_selected,dims=1)[1,:];
    if r>=3
        ylim_min = 0.75*minimum(μs);
        ylim_max = 1.25*maximum(μs);
    else
        ylim_min = minimum(μs .- 2*σs);
        ylim_max = maximum(μs .+ 1.5*σs);
    end
    plt = plot(xs, μs, 
        yerr = σs,  # Error bars
        ribbon=σs,
        label = L"\textbf{MZMD}",  # Legend label
        lw = 2,  # Line width
        lc = :blue,  # Line color
        markershape = :star,  # Marker shape
        markersize = 4,  # Marker size
        # markercolor = :blue,  # Marker color
        # color="blue",
        legend = :topright,  # Legend position
        grid = true,  # Show grid
        framestyle = :box  # Frame style
    )
    title!(L"\textrm{Relative ~ Generalization ~ Errors}", titlefont=20)
    xlabel!(L"\textrm{Memory ~ terms}", xtickfontsize=14, xguidefontsize=16)
        if error_type=="mse"
            ylabel!(L"L_2 \textrm{~ Relative ~ Errors}", ytickfontsize=10, yguidefontsize=16)
        else error_type=="max"
            ylabel!(L"L_{\infty} \textrm{~ Relative ~ Errors}", ytickfontsize=10, yguidefontsize=16)
        end
    savefig(plt, "./figures/dominant_modes_test_error_amx0_$(error_type)_$(data_type)_UQ_over_k_vs_d_lines_dts$(dt_skip)_r$(rs)_t$(t)_re$(re_num).png")
    return plt
end

kidx = 1:nk
plt = plot_k_vs_d_lines_uq(nk_range[kidx], nd_range, mzmd_errs_samps[:,kidx,:,:])
display(plt)
