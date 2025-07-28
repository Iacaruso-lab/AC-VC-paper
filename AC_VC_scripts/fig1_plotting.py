import seaborn as sb
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
import AC_VC_scripts.figure_formatting as ff
import AC_VC_scripts.loading_functions as lf
import AC_VC_scripts.MAPseq_data_processing as mdp
import AC_VC_scripts.helper_functions as hf
from seaborn.palettes import color_palette
from scipy.stats import pearsonr
import itertools
from matplotlib.colors import LogNorm
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
from matplotlib.patches import Patch
import matplotlib.ticker as mticker
from pathlib import Path
from scipy.stats import ttest_rel


def plot_MAPseq_cubelet_Allen_corr(gen_parameters, what_plot, ax):
    to_plot = mdp.make_Allen_MAPseq_comp_table(
        mice=gen_parameters["MICE"], proj_path=gen_parameters["proj_path"]
    )
    font_size = gen_parameters["font_size"]
    sb.set(style="white")
    mouse_ids = to_plot["Mouse"].unique()
    purple_palette = color_palette("Purples", len(mouse_ids))
    sb.set_palette(purple_palette)
    sb.scatterplot(data=to_plot, x="Allen_Z_core", y="MAPseq_Z_score", hue="Mouse", s=1)
    sb.regplot(
        data=to_plot,
        x="Allen_Z_core",
        y="MAPseq_Z_score",
        scatter=False,
        color="grey",
        line_kws={"linewidth": 0.5},
    )
    corr, p = pearsonr(to_plot["Allen_Z_core"], to_plot["MAPseq_Z_score"])
    x_label = "Total projection density \n bulk GFP (Z score)"
    y_label = "Total cubelet counts \n MAPseq (Z score)"
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    ax.tick_params(
        axis="both",
        which="both",
        direction="out",
        length=3,
        width=0.5,
        labelsize=font_size,
    )
    mant, exp = f"{p:.{2}E}".split("E")
    symbol_add, exp_add = ff.convert_to_exp(num=p)
    add_text = f"r = {corr:.3f}\np = {int(float(mant))}x10$^{{{exp_add}}}$"
    plt.text(
        0.02,
        0.98,
        add_text,
        transform=plt.gca().transAxes,
        fontsize=font_size,
        verticalalignment="top",
    )
    ax.tick_params(
        axis="both",
        which="both",
        bottom=True,
        top=False,
        left=True,
        right=False,
        direction="out",
        length=4,
        width=0.6,
        labelsize=font_size,
    )
    ff.myPlotSettings_splitAxis(
        fig=what_plot, ax=ax, ytitle=y_label, xtitle=x_label, title="", mySize=font_size
    )


def plot_Allen_corr(gen_parameters, what_plot, ax):
    """plot fig s2c showing correlations between Allen anterograde tracing dataset projection patterns at MAPseq cubelet locations"""
    to_plot = mdp.make_Allen_MAPseq_comp_table(
        mice=gen_parameters["MICE"], proj_path=gen_parameters["proj_path"]
    )
    font_size = gen_parameters["font_size"]
    colors = ["blue", "orange", "green"]
    which_expt = [
        "Allen_expt_A_Z_score",
        "Allen_expt_B_Z_score",
        "Allen_expt_C_Z_score",
    ]
    expt_name = ["120491896", "116903230", "100149109"]
    legend_entries = ["Mean"]

    for idx, (A, B) in enumerate(itertools.combinations([0, 1, 2], 2)):
        sb.scatterplot(
            data=to_plot,
            x=which_expt[A],
            y=which_expt[B],
            marker="X",
            s=1,
            color=colors[idx],
        )
        corr, p = pearsonr(to_plot[which_expt[A]], to_plot[which_expt[B]])
        legend_entries.append(
            f"r={np.round(corr, 2)}, {expt_name[A]} vs {expt_name[B]}"
        )
    sb.regplot(
        data=to_plot,
        x="Allen_Z_core",
        y="Allen_Z_core",
        scatter=False,
        color="grey",
        line_kws={"linewidth": 0.5},
    )
    plt.legend(
        legend_entries,
        loc="lower right",
        fontsize=font_size,
        bbox_to_anchor=(1, -1.25),
        frameon=False,
    )
    x_label = "Projection density \n expt. X bulk GFP (Z score)"
    y_label = "Projection density \n expt. Y bulk GFP (Z score)"
    ax.tick_params(
        axis="both",
        which="both",
        bottom=True,
        top=False,
        left=True,
        right=False,
        direction="out",
        length=4,
        width=0.6,
        labelsize=font_size,
    )
    ff.myPlotSettings_splitAxis(
        fig=what_plot, ax=ax, ytitle=y_label, xtitle=x_label, title="", mySize=font_size
    )
    plt.show()


def plot_broad_area_clustermap(proj_path, gen_parameters):
    areas_to_look = [
        "Ipsi cortex",
        "Contra cortex",
        "Striatum",
        "IC",
        "SC",
        "Thalamus",
    ]
    combined_table = mdp.assess_broad_proj_patterns(
        proj_path=proj_path, mice=gen_parameters["MICE"]
    )
    font_size = gen_parameters["font_size"]
    sb.set(font_scale=1, style="white")
    combined_table_to_look = combined_table[areas_to_look]
    cluster = sb.clustermap(
        combined_table_to_look.T,
        metric="canberra",
        row_cluster=True,
        standard_scale=0,
        norm=LogNorm(),
        cmap="Purples",
        figsize=(2.8, 1.5),
        xticklabels=False,
        yticklabels=True,
        cbar_pos=(0.8, 0.12, 0.02, 0.65),
        cbar_kws={"label": "Normalised \n projection strength"},
    )
    cluster.ax_row_dendrogram.set_visible(False)
    cluster.ax_col_dendrogram.set_visible(False)
    cluster.ax_heatmap.yaxis.set_ticks_position("left")
    cluster.ax_heatmap.yaxis.set_label_position("left")
    cluster.ax_heatmap.set_xlabel("")

    for spine in cluster.ax_heatmap.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color("black")
    for spine in cluster.cax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color("black")
    cluster.cax.set_ylabel(
        "Normalised barcode count", fontsize=font_size, rotation=270, labelpad=5
    )

    cluster.fig.canvas.draw_idle()
    cb_width = 0.02
    gap = 0.01
    heat_pos = cluster.ax_heatmap.get_position()
    cluster.cax.set_position(
        [
            heat_pos.x1 + gap,
            heat_pos.y0,
            cb_width,
            heat_pos.height,
        ]
    )
    cluster.cax.tick_params(
        axis="y", which="both", labelsize=font_size, length=4, width=0.5
    )
    heat_pos = cluster.ax_heatmap.get_position()
    num_barcodes = len(combined_table)
    tick_positions = range(0, num_barcodes, 1000)
    tick_labels = [str(pos) for pos in tick_positions]
    cluster.ax_heatmap.set_xticks(tick_positions)
    cluster.ax_heatmap.set_xticklabels(tick_labels, rotation=0)
    cluster.ax_heatmap.tick_params(which="both", length=3, color="black", width=0.5)
    cluster.ax_heatmap.tick_params(axis="y", which="major", labelsize=font_size)
    cluster.ax_heatmap.tick_params(axis="x", which="major", labelsize=font_size)
    cluster.ax_heatmap.xaxis.set_ticks_position("top")
    cluster.ax_heatmap.xaxis.set_label_position("top")
    cluster.ax_heatmap.set_xlabel("Neurons", fontsize=font_size, labelpad=2.5)
    cluster.cax.yaxis.set_minor_locator(plt.NullLocator())
    return cluster


def plot_upper_lower_proj_types(gen_parameters, proj_path, fig, ax):
    """plot fig. 1e to view distribution across layers of projection types"""
    IT_neurons, PT_neurons, CT_neurons = mdp.identifiy_layer_spec_projections(
        proj_path=proj_path, mice=gen_parameters["MICE"]
    )
    layers = ["Upper", "Lower"]
    font_size = gen_parameters["font_size"]
    bar_width = 0.6
    ax.bar(
        layers,
        IT_neurons,
        label="IT",
        color="purple",
        width=bar_width,
        edgecolor="none",
    )
    ax.bar(
        layers,
        PT_neurons,
        bottom=IT_neurons,
        label="PT",
        color="darkturquoise",
        width=bar_width,
        edgecolor="none",
    )
    ax.bar(
        layers,
        CT_neurons,
        bottom=[i + j for i, j in zip(IT_neurons, PT_neurons)],
        label="CT",
        color="orchid",
        width=bar_width,
        edgecolor="none",
    )
    ax.tick_params(
        axis="both",
        which="both",
        bottom=True,
        top=False,
        left=True,
        right=False,
        direction="out",
        length=4,
        width=0.6,
        labelsize=font_size,
    )
    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(axis="both", labelsize=font_size)

    ax.legend(
        loc="center left", bbox_to_anchor=(1, 0.8), frameon=False, fontsize=font_size
    )
    ff.myPlotSettings_splitAxis(
        fig=fig,
        ax=ax,
        ytitle="Proportion of neurons",
        xtitle="Layers",
        title="",
        mySize=font_size,
    )
    plt.show()


def plot_allen_area_vs_MAPseq_area(gen_parameters, fig, ax):
    """plot relationship between allen antergrae tracing projections and MAPseq projections at area level"""
    mice = gen_parameters["MICE"]
    proj_path = gen_parameters["proj_path"]
    combined_dict = mdp.samples_to_areas(mice=mice, proj_path=proj_path)
    common_areas = mdp.get_common_columns(
        mice=mice, combined_dict=combined_dict, cortex=True
    )
    mean_all_areas_df = mdp.get_allen_area_means(common_areas)
    font_size = gen_parameters["font_size"]
    logged_all_cortical = np.log10(mean_all_areas_df.astype(float))
    allen_z_normalized = (
        logged_all_cortical.T - logged_all_cortical.mean(axis=1)
    ) / logged_all_cortical.std(axis=1)
    allen_z_normalized = allen_z_normalized.T
    mean_series_list = [
        combined_dict[m]["homogenous_across_cubelet"][common_areas].mean() for m in mice
    ]
    mean_all_MAPseq = pd.concat(mean_series_list, axis=1)
    logged_all_MAPseq = np.log10(mean_all_MAPseq.T)
    # remove AUDp since we're looking at projection strengths and have dropped source sites
    cols = [col for col in allen_z_normalized.columns if col != "AUDp"]
    logged_all_MAPseq = logged_all_MAPseq[cols]
    allen_z_normalized = allen_z_normalized[cols]
    y = mean_all_MAPseq.T[cols].mean()
    x = allen_z_normalized[cols].mean()
    xerr = allen_z_normalized[cols].std()
    y_log = logged_all_MAPseq.mean()
    err_log = logged_all_MAPseq.std()
    y_lower = 10 ** (
        y_log - err_log
    )  # backtransforming since log scale and otherswise asymmetric
    y_upper = 10 ** (
        y_log + err_log
    )  # backtransforming since log scale and otherswise asymmetric
    yerr_asym = np.vstack((y - y_lower, y_upper - y))
    ax.errorbar(
        x,
        y,
        xerr=xerr,
        yerr=yerr_asym,
        fmt="none",
        color="gray",
        linewidth=0.5,
        alpha=1,
    )
    sb.scatterplot(x=x, y=y, color="black", s=5, ax=ax)
    ax.set_yscale("log", base=10)
    slope, intercept = np.polyfit(
        x, np.log10(y), 1
    )  # linear fit in log-space so the line is straight on the log axis
    x_fit = np.linspace(x.min(), x.max(), 200)
    y_fit = 10 ** (slope * x_fit + intercept)  # back-transform for plotting
    ax.plot(x_fit, y_fit, color="black", linewidth=0.5)
    r, p = pearsonr(x, np.log10(y))
    mant, exp = f"{p:.{2}E}".split("E")
    add_text = f"r = {r:.3f}\np = {int(float(mant))}x10$^{{{exp}}}$"
    x_label = "Mean bulk GFP \n (Z score)"
    y_label = "Mean MAPseq \nprojection density"
    ax.text(0.02, 1.1, add_text, transform=ax.transAxes, va="top")
    plt.tick_params(
        axis="both",
        which="both",
        bottom=True,
        top=False,
        left=True,
        right=False,
        direction="out",
        length=4,
        width=0.6,
        labelsize=font_size,
    )
    ff.myPlotSettings_splitAxis(
        fig=fig,
        ax=ax,
        ytitle=y_label,
        xtitle=x_label,
        title="",
        mySize=gen_parameters["font_size"],
    )
    ax.set_yticks([1e-1, 1e-2, 1e-3])
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xticks([-2, 0, 2])


def plot_dif_approaches_correlations(gen_parameters, fig, axes):
    """plot for extended data fig. 2d-f comparing correlations between mice using different area assignment approaches"""
    proj_path = gen_parameters["proj_path"]
    mice = gen_parameters["MICE"]
    combined_mice_dict = mdp.get_all_area_assign_approaches(
        proj_path=proj_path, mice=mice
    )
    font_size = gen_parameters["font_size"]
    axes = axes.flatten()
    analysis_list = ["homog_across_cubelet", "homog_across_area", "area_is_main"]
    mcc = MouseConnectivityCache()
    structure_tree = mcc.get_structure_tree()
    rsp = mcc.get_reference_space()
    color_list = ["forestgreen", "steelblue", "orchid"]
    titles = ["Homogeneous across cubelet", "Homogeneous across area", "Area is main"]
    mouse_color_mapping = {
        "forestgreen": "FIAA45.6a vs FIAA45.6d",
        "steelblue": "FIAA45.6a vs FIAA55.4d",
        "orchid": "FIAA45.6d vs FIAA55.4d",
    }

    legend_handles = [
        Patch(color=color, label=label) for color, label in mouse_color_mapping.items()
    ]
    x_label = "Log$_{10}$(Normalised projection \ndensity) mouse X"
    y_label = "Log$_{10}$(Normalised projection \ndensity) mouse Y"
    for i, key in enumerate(analysis_list):
        ax = axes[i]
        x = 0
        corr_list = []
        for mouse_1, mouse_2 in itertools.combinations(mice, 2):
            common_columns = set(combined_mice_dict[mouse_1][key].columns).intersection(
                combined_mice_dict[mouse_2][key].columns
            )
            common_cols_cortex = []
            for col in common_columns:
                if col != "Contra":
                    structure = structure_tree.get_structures_by_acronym([col])
                    if 315 in structure[0]["structure_id_path"]:
                        common_cols_cortex.append(col)
                if col == "Contra":
                    common_cols_cortex.append(col)
            mean_A = np.log10(
                combined_mice_dict[mouse_1][key][common_cols_cortex].mean(axis=0)
                + 1e-10
            )
            mean_B = np.log10(
                combined_mice_dict[mouse_2][key][common_cols_cortex].mean(axis=0)
                + 1e-10
            )
            corr, p = pearsonr(mean_A, mean_B)
            print(mouse_1, mouse_2, corr)
            corr_list.append(corr)
            sb.regplot(
                x=mean_A,
                y=mean_B,
                ax=ax,
                scatter=True,
                label=f"r = {np.round(corr, 2)}",
                scatter_kws={"color": color_list[x], "s": 1, "alpha": 0.5},
                line_kws={
                    "color": color_list[x],
                    "linewidth": 0.5,
                },
            )
            x = x + 1

        ax.tick_params(
            axis="both",
            which="both",
            bottom=True,
            top=False,
            left=True,
            right=False,
            direction="out",
            length=4,
            width=0.6,
            labelsize=font_size,
        )
        ax.legend(
            loc="upper left",
            fontsize=font_size,
            frameon=False,
            bbox_to_anchor=(-0.1, 1.1),
        )
        ff.myPlotSettings_splitAxis(
            fig=fig,
            ax=ax,
            ytitle=y_label,
            xtitle=x_label,
            title=titles[i],
            mySize=font_size,
        )
        print(np.mean(corr_list), key)
    axes[3].axis("off")
    fig.legend(
        handles=legend_handles,
        loc="lower right",
        bbox_to_anchor=(0.9, 0.25),
        fontsize=font_size,
        frameon=False,
    )
    plt.tight_layout()
    plt.show()


def plot_within_out_AUD_proj_frequencies(gen_parameters, fig, ax):
    """plot projection frequenceis with and outside the auditory cortex"""

    font_size = gen_parameters["font_size"]
    mice = gen_parameters["MICE"]
    proj_path = gen_parameters["proj_path"]
    count_df = pd.DataFrame(
        index=mice,
        columns=[
            "Total",
            "Only Within Auditory Cortex",
            "Out of Auditory Cortex",
        ],
    )
    for num, mouse in enumerate(mice):
        parameters_path = f"{proj_path}/{mouse}/Sequencing"
        barcodes = pd.read_pickle(
            f"{parameters_path}/A1_barcodes_thresholded_with_source.pkl"
        )
        bc_no_source = pd.read_pickle(f"{parameters_path}/A1_barcodes_thresholded.pkl")
        bc_no_source = bc_no_source[bc_no_source.sum(axis=1) > 0]
        tot_bc = len(barcodes)
        count_df.loc[mouse, "Total"] = tot_bc
        lcm_directory = Path(f"{proj_path}/{mouse}/LCM")
        areas_only_grouped = mdp.get_area_volumes(
            barcode_table_cols=barcodes.columns,
            lcm_directory=lcm_directory,
            area_threshold=0.1,
        )
        with_aud = areas_only_grouped[
            (areas_only_grouped[["AUDp", "AUDv", "AUDd", "AUDpo"]] > 0).any(axis=1)
        ].index.to_list()
        with_aud = [sample for sample in with_aud if sample in barcodes.columns]
        no_AUD_mat = barcodes.drop(columns=with_aud)
        no_AUD_mat = no_AUD_mat[no_AUD_mat.sum(axis=1) > 0]
        count_df.loc[mouse, "Only Within Auditory Cortex"] = tot_bc - len(no_AUD_mat)
        count_df.loc[mouse, "Out of Auditory Cortex"] = len(no_AUD_mat)
    freq_columns = ["Only Within Auditory Cortex", "Out of Auditory Cortex"]
    frequencies = count_df[freq_columns].div(count_df["Total"], axis=0)
    mean_frequencies = frequencies.mean()
    x = np.arange(len(freq_columns))
    bar_width = 0.6
    bar_colors = ["#6d6e71", "#d1d3d4"]
    ax.bar(
        x,
        mean_frequencies,
        width=bar_width,
        label="Mean Frequency",
        color=bar_colors,
        linewidth=0.5,
        zorder=1,
    )

    for i, column in enumerate(freq_columns):
        y = frequencies[column]
        ax.scatter(
            [x[i]] * len(y),
            y,
            label=f"{column} Replicates" if i == 0 else "",
            color="grey",
            zorder=2,
            s=5,
        )

    wrapped_labels = ["Only\nAC", "Out of\nAC"]
    ax.set_xticks(x)
    ax.set_xticklabels(wrapped_labels, fontsize=font_size)

    y_label = "Proportion of neurons"
    plt.yticks(size=font_size)
    ff.myPlotSettings_splitAxis(
        fig=fig, ax=ax, ytitle=y_label, xtitle="", title="", mySize=font_size
    )
    ax.tick_params(
        axis="both",
        which="both",
        bottom=True,
        top=False,
        left=True,
        right=False,
        direction="out",
        length=4,
        width=0.6,
        labelsize=font_size,
    )
    ax.set_xticklabels(wrapped_labels, fontsize=font_size)
    ax.set_yticks(ax.get_yticks())
    ax.set_yticklabels([f"{tick:.2f}" for tick in ax.get_yticks()], fontsize=font_size)


def plot_overall_proj_frequencies_areas(gen_parameters, fig, ax):
    """fig. 1i plotting"""
    font_size = gen_parameters["font_size"]
    mice = gen_parameters["MICE"]
    proj_path = gen_parameters["proj_path"]
    combined_dict = mdp.samples_to_areas(mice=mice, proj_path=proj_path)
    convert_dict = hf.get_convert_dict()
    all_mice_combined = pd.concat(
        [
            combined_dict[k]["homogenous_across_cubelet"][
                mdp.get_common_columns(
                    mice=mice, combined_dict=combined_dict, cortex=True
                )
            ]
            for k in mice
        ]
    )
    area_groups = {
        "AUDITORY CORTEX": ["AUDd", "AUDv", "AUDpo"],
        "LATERAL CORTEX": ["TEa", "PERI", "ECT"],
        "VISUAL CORTEX": [
            "VISal",
            "VISp",
            "VISpl",
            "VISpm",
            "VISrl",
            "VISpor",
            "VISam",
            "VISl",
            "VISa",
            "VISli",
        ],
        "SOMATOMOTOR CORTICAL AREAS": ["MOp", "MOs", "SSp", "SSs"],
        "CINGULATE CORTEX": ["RSPv", "RSPd", "RSPagl", "ACAd", "ACAv"],
    }
    common_cols_cortex = all_mice_combined.columns
    freq_df = pd.DataFrame(index=mice, columns=common_cols_cortex)
    tot_freq = pd.DataFrame(index=mice, columns=area_groups.keys())
    for mouse in mice:
        df_to_look = (
            combined_dict[mouse]["homogenous_across_cubelet"].astype(bool).astype(int)
        )
        for area in common_cols_cortex:
            freq_df.loc[mouse, area] = df_to_look[area].mean()
        for grouped_area in tot_freq.columns:
            frequency = len(
                df_to_look[df_to_look[area_groups[grouped_area]].sum(axis=1) > 0]
            ) / len(df_to_look)
            tot_freq.loc[mouse, grouped_area] = frequency

    plot_data = []
    group_positions = []
    x_labels = []
    x_pos = 0
    for group, columns in area_groups.items():
        sorted_columns = sorted(
            columns, key=lambda col: freq_df[col].mean(), reverse=True
        )
        group_mean = tot_freq[group]
        plot_data.append(group_mean)
        x_labels.append(f"Total")
        group_positions.append(x_pos - 0.5)
        x_pos += 1
        group_positions.append(x_pos - 0.5)

        for area in sorted_columns:
            plot_data.append(freq_df[area])
            x_labels.append(area)
            x_pos += 1

    x_labels = [
        item if item not in convert_dict.keys() else convert_dict[item]
        for item in x_labels
    ]
    plot_df = pd.concat(plot_data, axis=1).T
    plot_df.columns = freq_df.index
    plot_df.index = x_labels

    for i, (label, row) in enumerate(plot_df.iterrows()):
        is_total = label == "Total" or label.startswith("Total")
        bar_color = "black" if is_total else "mediumpurple"

        ax.scatter(
            [i] * len(row),
            row,
            color="grey",
            edgecolor="grey",
            s=5,
            zorder=2,
            label="Individual Points" if i == 0 else "",
        )
        ax.bar(
            i,
            row.mean(),
            color=bar_color,
            zorder=1,
            width=0.75,
            label="Mean" if i == 0 else "",
        )
    ax.set_xticks(range(len(plot_df.index)))

    ax.tick_params(axis="y", labelsize=font_size, length=4, width=0.5)
    ff.myPlotSettings_splitAxis(
        fig=fig,
        ax=ax,
        ytitle="Proportion of neurons",
        xtitle="Cortical areas",
        title="",
        mySize=font_size,
    )

    total_indices = [
        i
        for i, label in enumerate(plot_df.index)
        if label == "Total" or label.startswith("Total")
    ]

    for idx in total_indices:
        ax.axvline(x=idx - 0.5, color="k", linestyle="--", linewidth=0.5, zorder=0)

    ax.axvline(
        x=len(plot_df.index) - 0.5, color="k", linestyle="--", linewidth=0.5, zorder=0
    )

    ax.tick_params(
        axis="both",
        which="both",
        bottom=True,
        top=False,
        left=True,
        right=False,
        direction="out",
        length=4,
        width=0.6,
        labelsize=font_size,
    )
    ax.set_xticklabels(plot_df.index, rotation=90, ha="center", fontsize=font_size)

    ax.margins(x=0)
    ax.set_xlim(-0.5, len(plot_df.index) - 0.5)
    ax.xaxis.set_label_coords(0.5, -0.6)


def plot_negative_control_UMI_counts(fig, axes, gen_parameters):
    negs_dict = mdp.identify_neg_UMI_counts(gen_parameters=gen_parameters)
    font_size = gen_parameters["font_size"]
    dist_list = []
    for mouse, df in negs_dict.items():
        dist = df.loc[df["barcode_counts"] > 0, "barcode_counts"].value_counts()
        dist_list.append(dist)
    total_distribution = pd.concat(dist_list, axis=1).fillna(0).sum(axis=1)
    total_distribution = total_distribution.sort_index()
    total_distribution_norm = total_distribution / total_distribution.sum()
    axes.bar(
        total_distribution_norm.index,
        total_distribution_norm.values,
        width=1,
        color="black",
    )
    axes.axvline(x=1.5, color="red", linestyle="dotted", linewidth=1, label="cut off")
    plt.legend(
        loc="upper right",
        bbox_to_anchor=(1.2, 1),
        frameon=False,
    )
    plt.xticks(rotation=90)
    ff.myPlotSettings_splitAxis(
        fig=fig,
        ax=axes,
        ytitle="Frequency",
        xtitle=f"UMI Counts\nNegative control",
        title="",
        mySize=font_size,
    )
    axes.xaxis.set_major_locator(mticker.MultipleLocator(1))
    axes.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x)}"))
    plt.show()


def plot_sssynth_res(gen_parameters, fig, axes):
    """function to plot extended data fig. 1b difference in second strand synthesis efficiency using targeted versus non-targeted approaches"""
    font_size = gen_parameters["font_size"]
    proj_path = gen_parameters["proj_path"]
    ssynth = pd.read_excel(
        f"{proj_path}/supplementary_tables/second_str_synth_results_120122.xlsx"
    )  # load excel file recording results of non-targeted versus targeted second strand synthesis approach
    non_targeted = ssynth["Non-targeted"]
    targeted = ssynth["Targeted"]
    t_stat, p_value_two_tailed = ttest_rel(targeted, non_targeted)
    if t_stat > 0:
        p_value_one_tailed = p_value_two_tailed / 2
    else:
        p_value_one_tailed = 1 - (p_value_two_tailed / 2)

    print(
        f"Paired t-test result: t-statistic = {t_stat:.4f}, p-value = {p_value_two_tailed:.4f}"
    )
    x_positions = [1, 2]
    for i in range(len(non_targeted)):
        plt.plot(
            x_positions,
            [non_targeted[i], targeted[i]],
            color="black",
            marker="o",
            markersize=0.2,
            linewidth=0.3,
        )
    plt.scatter(
        np.ones(len(non_targeted)),
        non_targeted,
        color="black",
        label="Non-Targeted",
        s=1,
    )
    plt.scatter(
        np.ones(len(targeted)) * 2, targeted, color="black", label="Targeted", s=1
    )
    plt.xticks([1, 2], ["Non-Targeted", "Targeted"])
    if p_value_one_tailed < 0.01:
        max_y = max(max(non_targeted), max(targeted))
        plt.text(1.5, max_y * 0.8, "**", fontsize=10, ha="center")
    plt.ylim(0, max_y * 1.1)
    plt.xticks([1.1, 1.9], ["Non-Targeted", "Targeted"])
    ff.myPlotSettings_splitAxis(
        fig=fig,
        ax=axes,
        ytitle="Concentration (ng/μL)",
        xtitle="",
        title="",
        mySize=font_size,
    )
    plt.show()


def plot_entropy_AT_content_2d_hist(fig, ax_entropy, ax_at, gen_parameters):
    """plot 2d histograms showing template switching event analysis in extended data fig 1d"""
    proj_path = gen_parameters["proj_path"]
    template_switches = hf.analyse_template_switches(proj_path=proj_path)
    y_bins = np.arange(1, 100, 4)
    hist = ax_entropy.hist2d(
        template_switches["entropy"],
        template_switches["relative_abundance"],
        norm=LogNorm(),
        bins=(20, y_bins),
    )
    cbar = fig.colorbar(hist[3], ax=ax_entropy, label="# Neurons")
    cbar.ax.yaxis.label.set_rotation(270)
    ax_entropy.set_xlabel("Entropy of UMI sequence")
    ax_entropy.set_ylabel("1st/2nd abundance")
    template_switches["bin"] = np.digitize(
        template_switches["relative_abundance"], y_bins
    )
    grouped = template_switches.groupby("bin").agg(
        count=("entropy", "size"), mean_entropy=("entropy", "mean")
    )
    means_to_plot, y_bin_centers = [], []
    for bin_index, row in grouped.iterrows():
        if row["count"] > 5:
            y_center = y_bins[bin_index - 1] + np.diff(y_bins)[0] / 2
            means_to_plot.append(row["mean_entropy"])
            y_bin_centers.append(y_center)

    ax_entropy.plot(
        means_to_plot,
        y_bin_centers,
        color="deeppink",
        markersize=0.5,
        marker="o",
        linestyle="--",
        label="Mean Entropy",
    )
    ax_entropy.legend(loc="upper left", frameon=True)

    hist = ax_at.hist2d(
        template_switches["AT_content"],
        template_switches["relative_abundance"],
        norm=LogNorm(),
        bins=(10, y_bins),
    )
    cbar = fig.colorbar(hist[3], ax=ax_at, label="# Neurons")
    cbar.ax.yaxis.label.set_rotation(270)

    ax_at.set_xlabel("UMI sequence A/T content")

    template_switches["bin"] = np.digitize(
        template_switches["relative_abundance"], y_bins
    )
    grouped = template_switches.groupby("bin").agg(
        count=("AT_content", "size"), mean_AT_content=("AT_content", "mean")
    )
    means_to_plot, y_bin_centers = [], []
    for bin_index, row in grouped.iterrows():
        if row["count"] > 5:
            y_center = y_bins[bin_index - 1] + np.diff(y_bins)[0] / 2
            means_to_plot.append(row["mean_AT_content"])
            y_bin_centers.append(y_center)

    ax_at.plot(
        means_to_plot,
        y_bin_centers,
        color="deeppink",
        markersize=0.5,
        marker="o",
        linestyle="--",
        label="Mean A/T content",
    )
    ax_at.legend(loc="upper left", frameon=True)

    for ax in (ax_entropy, ax_at):
        ax.tick_params(axis="both", which="both", direction="out", length=4, width=0.6)

    fig.tight_layout()
    plt.show()
