include("cluster.jl")
function fission_gaussian(X::AbstractMatrix, Σ::AbstractMatrix, τ::Real; q = 0.05, cl = nothing, ret_cl = false)
    n, p = size(X)
    Z = rand(MvNormal(Σ), n)'
    fX = X + τ * Z
    gX = X - 1/τ * Z

    # clustering on fX
    #cl = kmeans(fX', 2)
    #gX1 = gX[cl.assignments .== 1, :]
    #gX2 = gX[cl.assignments .== 2, :]
    if isnothing(cl)
        cl = rcopy(R"kmeans($fX, 2)$cluster")
        if ret_cl
            return cl
        end
    end
    gX1 = gX[cl .== 1, :]
    gX2 = gX[cl .== 2, :]

    pvals = zeros(p)
    for i = 1:p
        pvals[i] = rcopy(R"t.test($(gX1[:, i]), $(gX2[:, i]))$p.value")
    end
    padj = R"p.adjust"(pvals, method = "BH")
    return findall(padj .< q)
end

function fission_gaussian(X::AbstractMatrix, Σ::AbstractMatrix, τs::AbstractVector, truth::AbstractVector; q = 0.05, cl = nothing)
    nτ = length(τs)
    fdrs = zeros(nτ)
    powers = zeros(nτ)
    for i = 1:nτ
        sel = fission_gaussian(X, Σ, τs[i], cl = cl)
        fdr, f1, prec, recall = calc_acc(sel, truth)
        fdrs[i] = fdr
        powers[i] = recall
    end
    return fdrs, powers
end

function experiment_fission_with_ds_dd(; δ = 0.6, ρ = 0.0, sigma = 0.0, fig = true, τs = 0.1:0.3:4.0, oracle = false)
    n = 1000
    p = 2000
    #ρ = 0.0
    #δ = 0.6
    Σ = ar1(ρ, p)
    prop_imp = 0.1
    x, cl = gen_data_normal(n, p, δ, prop_imp = prop_imp, ρ = ρ, corr_structure = "ar1", sigma = sigma)
    truth = Int.(1:p*prop_imp)
    #τs = 0.1:0.3:4.0    
    fdrs, powers = fission_gaussian(x, Σ, τs, truth, cl = ifelse(oracle, cl, nothing))
    sel_dd = double_dipping(x)
    fdr_dd, _, _, power_dd = calc_acc(sel_dd, truth)
    sel_ds = mds2(x, method = "tstat", M = 1)
    fdr_ds, _, _, power_ds = calc_acc(sel_ds, truth)
    sel_mds = mds2(x, method = "tstat", M = 10)
    fdr_mds, _, _, power_mds = calc_acc(sel_mds, truth)
    if fig
        plot(τs, fdrs, xlab = "τ", label = "FDR", markershape = :x, 
                title = latexstring("\$n = $n, p = $p, \\rho = $ρ, \\delta = $δ, \\sigma = $sigma\$"), 
                legend = :topright, ylim = (0, 1))
        hline!([0.05], label = "", ls = :dash)
        hline!([fdr_mds], label = "FDR (MDS)", ls = :dot)
        hline!([fdr_ds], label = "FDR (DS)", ls = :dot)
        hline!([fdr_dd], label = "FDR (DD)", ls = :dot)
        plot!(τs, powers, label = "Power", markershape = :x)
        hline!([power_mds], label = "Power (MDS)")
        hline!([power_ds], label = "Power (DS)")
        hline!([power_dd], label = "Power (DD)")
    else
        return fdrs, powers, [fdr_dd, fdr_ds, fdr_mds], [power_dd, power_ds, power_mds]
    end
end

function experiment_fission(; δ = 0.6, ρ = 0.0, sigma = 0.0, fig = true, τs = 0.1:0.3:4.0, oracle = false)
    n = 1000
    p = 2000
    #ρ = 0.0
    #δ = 0.6
    Σ = ar1(ρ, p)
    prop_imp = 0.1
    x, cl = gen_data_normal(n, p, δ, prop_imp = prop_imp, ρ = ρ, corr_structure = "ar1", sigma = sigma)
    truth = Int.(1:p*prop_imp)
    #τs = 0.1:0.3:4.0    
    fdrs, powers = fission_gaussian(x, Σ, τs, cl = ifelse(oracle, cl, nothing))

    if fig
        plot(τs, fdrs, xlab = "τ", label = "FDR", markershape = :x, title = latexstring("\$n = $n, p = $p, \\rho = $ρ, \\delta = $δ, \\sigma = $sigma\$"), legend = :topright)
        hline!([0.05], label = "", ls = :dash)
        plot!(τs, powers, label = "Power", markershape = :x)
    else
        return fdrs, powers
    end
end

function run_fission_ds_dd(; nrep = 10, σ = 0.1, ρ = 0.1, δ = 0.6)
    resfile = "../res/data-fission/fission-ds-dd-sigma$σ-delta$δ-rho$ρ-nrep$nrep.sil"
    res = pmap(x -> experiment_fission_with_ds_dd(δ = δ, ρ = ρ, sigma = 0.1, fig = false, τs = 0.1:0.3:4.0), 1:nrep)
    serialize(resfile, res)
    return res
end

function summary_fission_dd_ds_along_signal(ρ = 0.0)
    δs = 0.1:0.1:0.6
    nδ = length(δs)
    # Fission, DS, MDS, DD
    nmethod = 4
    fdr = zeros(nmethod, nδ)
    power = zeros(nmethod, nδ)
    for (i, δ) in enumerate(δs)
        resfile = "../res/data-fission/fission-ds-dd-sigma0.1-delta$δ-rho$ρ-nrep100.sil"
        res = deserialize(resfile)
        μfdr_fission = mean(hcat([x[1] for x in res]...), dims = 2)
        μpower_fission = mean(hcat([x[2] for x in res]...), dims = 2)
        μfdr_ddds = mean(hcat([x[3] for x in res]...), dims = 2)
        μpower_ddds = mean(hcat([x[4] for x in res]...), dims = 2)
        idx = argmax(μpower_fission)
        power[1, i] = μpower_fission[idx]
        fdr[1, i] = μfdr_fission[idx]
        power[2:4, i] .= μpower_ddds
        fdr[2:4, i] .= μfdr_ddds
    end
    #return fdr, power
    #plot_fdr_and_power_vs_signal(δs, fdr, power, ["DF" "DD" "DS" "MDS"], [:star6 :dtriangle :circle :rect], "Gaussian (ρ = $ρ)", nothing, nothing, offset = 0, alphas = [1, 1, 0.6, 0.6], plot_even_height = true)
    plot_fdr_and_power_vs_signal(δs, fdr, power, ["DF" "DD" "DS" "MDS"], [:rect :circle :diamond :star6], "Gaussian (ρ = $ρ)", nothing, nothing, offset = 0, alphas = [1, 1, 1, 1], plot_even_height = true)
    #plot_fdr_and_power_vs_signal(δs, fdr[[1,2,4], :], power[[1,2,4], :], ["DF" "DD" "MDS"], [:x :dtriangle :star5], "ρ = $ρ", nothing, nothing)
end

function summary_fission_poisson()
    resfile = "../res/jl-cluster/2024-08-31T22_10_50-04_00-poisson-M10-nrep100-n1000-p2000-prop0.1-nsig6-0.05-0.3-rho0-bs2000-sigma0.1-mds2.sil"
    RES, δs = deserialize(resfile)
    nδ = length(δs)
    # DD, CountSplit/Data Fission, MDS, DS
    nmethod = 4
    fdr = zeros(nmethod, nδ)
    power = zeros(nmethod, nδ)
    for (i, δ) in enumerate(δs)
        μfdr_fission = mean(RES[i][:, 4+1])
        μpower_fission = mean(RES[i][:, 4+4])
        μfdr_dd = mean(RES[i][:, 1])
        μpower_dd = mean(RES[i][:, 4])
        μfdr_mds = mean(RES[i][:, 8+1])
        μpower_mds = mean(RES[i][:, 8+4])
        μfdr_ds = mean([x[1] for x in RES[i][:, 13] ])
        μpower_ds = mean([x[4] for x in RES[i][:, 13]])

        fdr[:, i] .= [μfdr_fission, μfdr_dd, μfdr_ds, μfdr_mds]
        power[:, i] .= [μpower_fission, μpower_dd, μpower_ds, μpower_mds]
    end
    fig = plot_fdr_and_power_vs_signal(δs, fdr, power, ["DF" "DD" "DS" "MDS"], [:rect :circle :diamond :star6], "Poisson", nothing, nothing, offset = 0, alphas = [1, 1, 1, 1], plot_even_height = true)
    savefig(fig, "../res/offset0-4fdr_and_power_along_signal-fission-ds-dd-poisson0.pdf")
end

function multi_summary_along_signal()
    ρs = 0.0:0.1:0.9
    nρ = length(ρs)
    for i = 1:nρ
        fig = summary_fission_dd_ds_along_signal(ρs[i])
        savefig(fig, "../res/data-fission/offset0-4fdr_and_power_along_signal-fission-ds-dd-rho$(ρs[i]).pdf")
    end
end
