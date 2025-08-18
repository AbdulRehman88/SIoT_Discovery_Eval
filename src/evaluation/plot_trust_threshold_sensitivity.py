# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os
#
# # Set styles
# sns.set(style="whitegrid")
# plt.rcParams.update({
#     "font.size": 12,
#     "font.family": "serif",
#     "figure.figsize": (10, 6)
# })
#
# # Load data
# input_csv = r"D:\Dr_Abdul_Rehman\MyPycharm\August 02 2025\SIoT_Discovery_Eval\results\trust_threshold_sensitivity.csv"
# df = pd.read_csv(input_csv)
#
# # Create output folder
# output_dir = os.path.join(os.path.dirname(input_csv), "plots")
# os.makedirs(output_dir, exist_ok=True)
#
# def plot_metric(df, y_col, y_label, filename):
#     plt.figure()
#     sns.lineplot(data=df, x="Alpha", y=y_col, hue="Dataset", marker="o", linewidth=2)
#     plt.ylabel(y_label)
#     plt.title(f"{y_label} vs Trust Threshold α")
#     plt.xlabel("Trust Threshold α")
#     plt.xticks(sorted(df['Alpha'].unique()))
#     plt.legend(title="Dataset", loc="best")
#     plt.tight_layout()
#     save_path = os.path.join(output_dir, filename)
#     plt.savefig(save_path, dpi=300)
#     plt.close()
#     print(f"✅ Saved: {save_path}")
#
# # Plot each metric
# plot_metric(df, "SuccessRate", "Success Rate", "success_rate_vs_alpha.png")
# plot_metric(df, "AvgHops", "Average Hops", "avg_hops_vs_alpha.png")
# plot_metric(df, "AvgTime", "Average Time (s)", "avg_time_vs_alpha.png")



#===================================== New code with both greedy and fallback ============================


#
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os
#
# # Set styles
# sns.set(style="whitegrid")
# plt.rcParams.update({
#     "font.size": 12,
#     "font.family": "serif",
#     "figure.figsize": (10, 6)
# })
#
# # Load CSVs
# greedy_csv = r"D:\Dr_Abdul_Rehman\MyPycharm\August 02 2025\SIoT_Discovery_Eval\results\trust_threshold_sensitivity_top6_trials100_greedy.csv"
# fallback_csv = r"D:\Dr_Abdul_Rehman\MyPycharm\August 02 2025\SIoT_Discovery_Eval\results\trust_threshold_sensitivity_top6_trials100_fallback.csv"
#
# df_greedy = pd.read_csv(greedy_csv)
# df_greedy["Strategy"] = "Greedy"
#
# df_fallback = pd.read_csv(fallback_csv)
# df_fallback["Strategy"] = "Fallback"
#
# # Combine into one DataFrame
# df_combined = pd.concat([df_greedy, df_fallback], ignore_index=True)
#
# # Output folder
# output_dir = os.path.join(os.path.dirname(greedy_csv), "plots")
# os.makedirs(output_dir, exist_ok=True)
#
# def plot_metric(df, y_col, y_label, filename):
#     plt.figure()
#
#     # Define line styles
#     line_styles = {"Greedy": "-", "Fallback": "--"}
#     datasets = df["Dataset"].unique()
#     palette = sns.color_palette("tab10", n_colors=len(datasets))
#
#     for i, dataset in enumerate(datasets):
#         for strategy in ["Greedy", "Fallback"]:
#             subset = df[(df["Dataset"] == dataset) & (df["Strategy"] == strategy)]
#             if subset.empty:
#                 continue
#             linestyle = line_styles[strategy]
#             label = f"{dataset} ({strategy})"
#             plt.plot(
#                 subset["Alpha"],
#                 subset[y_col],
#                 label=label,
#                 linestyle=linestyle,
#                 color=palette[i],
#                 marker="o"
#             )
#
#     plt.xlabel("Trust Threshold α")
#     plt.ylabel(y_label)
#     plt.title(f"{y_label} vs Trust Threshold α")
#     plt.legend(loc="best", ncol=2)
#     plt.tight_layout()
#     save_path = os.path.join(output_dir, filename)
#     plt.savefig(save_path, dpi=300)
#     plt.close()
#     print(f"✅ Saved: {save_path}")
#
# # Plot each metric
# plot_metric(df_combined, "SuccessRate", "Success Rate", "combined_success_rate_vs_alpha.png")
# plot_metric(df_combined, "AvgHops", "Average Hops", "combined_avg_hops_vs_alpha.png")
# plot_metric(df_combined, "AvgTime", "Average Time (s)", "combined_avg_time_vs_alpha.png")




#====================== New code with latest results 09-08-2025 (finally updated results. ) ===========================================



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set styles
sns.set(style="whitegrid")
plt.rcParams.update({
    "font.size": 12,
    "font.family": "serif",
    "figure.figsize": (10, 6)
})

# Load CSVs
greedy_csv = r"D:\Dr_Abdul_Rehman\MyPycharm\August 02 2025\SIoT_Discovery_Eval\results\trust_threshold_sensitivity_top6_trials100_greedy.csv"
fallback_csv = r"D:\Dr_Abdul_Rehman\MyPycharm\August 02 2025\SIoT_Discovery_Eval\results\trust_threshold_sensitivity_top6_trials100_fallback.csv"

df_greedy = pd.read_csv(greedy_csv)
df_greedy["Strategy"] = "Greedy"

df_fallback = pd.read_csv(fallback_csv)
df_fallback["Strategy"] = "Fallback"

# Combine into one DataFrame
df_combined = pd.concat([df_greedy, df_fallback], ignore_index=True)

# Sort within each (Dataset, Strategy) by Alpha so lines are drawn in order
df_combined = df_combined.sort_values(by=["Dataset", "Strategy", "Alpha"]).reset_index(drop=True)

# Output folder
output_dir = os.path.join(os.path.dirname(greedy_csv), "plots")
os.makedirs(output_dir, exist_ok=True)

def _save_all_formats(basepath_no_ext, dpi=600):
    """Save current figure as PNG, JPG, and PDF with the requested DPI."""
    png_path = basepath_no_ext + ".png"
    jpg_path = basepath_no_ext + ".jpg"
    pdf_path = basepath_no_ext + ".pdf"
    plt.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.savefig(jpg_path, dpi=dpi, bbox_inches="tight")
    plt.savefig(pdf_path, dpi=dpi, bbox_inches="tight")
    print(f"✅ Saved: {png_path}")
    print(f"✅ Saved: {jpg_path}")
    print(f"✅ Saved: {pdf_path}")

def plot_metric(df, y_col, y_label, filename_stem):
    plt.figure()

    # Define line styles
    line_styles = {"Greedy": "-", "Fallback": "--"}
    datasets = df["Dataset"].unique()
    palette = sns.color_palette("tab10", n_colors=len(datasets))

    for i, dataset in enumerate(datasets):
        for strategy in ["Greedy", "Fallback"]:
            subset = df[(df["Dataset"] == dataset) & (df["Strategy"] == strategy)]
            if subset.empty:
                continue
            linestyle = line_styles.get(strategy, "-")
            label = f"{dataset} ({strategy})"
            plt.plot(
                subset["Alpha"],
                subset[y_col],
                label=label,
                linestyle=linestyle,
                color=palette[i],
                marker="o"
            )

    plt.xlabel("Trust Threshold α")
    plt.ylabel(y_label)
    plt.title(f"{y_label} vs Trust Threshold α")
    plt.legend(loc="best", ncol=2)
    plt.tight_layout()

    basepath = os.path.join(output_dir, filename_stem)
    _save_all_formats(basepath, dpi=600)
    plt.close()

# Existing plots (kept as-is, now saved in PNG/JPG/PDF @600dpi)
plot_metric(df_combined, "SuccessRate", "Success Rate", "combined_success_rate_vs_alpha")
plot_metric(df_combined, "AvgHops", "Average Hops", "combined_avg_hops_vs_alpha")
plot_metric(df_combined, "AvgTime", "Average Time (s)", "combined_avg_time_vs_alpha")

# NEW: AvgVisited vs α (Greedy vs Fallback)
plot_metric(df_combined, "AvgVisited", "Average Nodes Visited", "combined_avg_visited_vs_alpha")

# NEW: LCCRatio vs α (connectivity reasoning)
# (Identical between Greedy/Fallback, but we’ll still plot both for completeness.)
plot_metric(df_combined, "LCCRatio", "Largest Connected Component Ratio", "combined_lcc_ratio_vs_alpha")
