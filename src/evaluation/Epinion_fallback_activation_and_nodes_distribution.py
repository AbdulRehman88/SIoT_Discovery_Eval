import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Paths to your CSVs
greedy_csv = r"D:\Dr_Abdul_Rehman\MyPycharm\August 02 2025\SIoT_Discovery_Eval\results\trust_threshold_sensitivity_top6_trials100_greedy.csv"
fallback_csv = r"D:\Dr_Abdul_Rehman\MyPycharm\August 02 2025\SIoT_Discovery_Eval\results\trust_threshold_sensitivity_top6_trials100_fallback.csv"

# Load data
df_greedy = pd.read_csv(greedy_csv)
df_fallback = pd.read_csv(fallback_csv)

# Output folder for plots
output_dir = os.path.join(os.path.dirname(greedy_csv), "plots")
os.makedirs(output_dir, exist_ok=True)

# --------------------------
# 1. Fallback Activation Rate Table (Epinions)
# --------------------------
epinions_fallback = df_fallback[df_fallback["Dataset"] == "Epinions"].copy()
if "FallbackUsed" in epinions_fallback.columns and "Trials" in epinions_fallback.columns:
    epinions_fallback["FallbackActivationRate (%)"] = (epinions_fallback["FallbackUsed"] / epinions_fallback["Trials"]) * 100
    rate_table = epinions_fallback[["Alpha", "FallbackActivationRate (%)"]]
    print("\n📊 Fallback Activation Rate (Epinions):")
    print(rate_table.to_string(index=False))
    # Save table to CSV
    rate_table.to_csv(os.path.join(output_dir, "epinions_fallback_activation_rate.csv"), index=False)
else:
    print("⚠️ 'FallbackUsed' or 'Trials' columns not found in fallback CSV.")

# --------------------------
# 2. Visited Nodes Distribution Plot (Epinions)
# --------------------------
sns.set(style="whitegrid")
plt.figure(figsize=(8, 5))
line_styles = {"Greedy": "-", "Fallback": "--"}

# Greedy
greedy_subset = df_greedy[df_greedy["Dataset"] == "Epinions"]
plt.plot(greedy_subset["Alpha"], greedy_subset["AvgVisited"],
         label="Epinions (Greedy)", linestyle=line_styles["Greedy"],
         marker="o", color="blue")

# Fallback
fallback_subset = df_fallback[df_fallback["Dataset"] == "Epinions"]
plt.plot(fallback_subset["Alpha"], fallback_subset["AvgVisited"],
         label="Epinions (Fallback)", linestyle=line_styles["Fallback"],
         marker="o", color="red")

plt.xlabel("Trust Threshold α")
plt.ylabel("Average Nodes Visited")
plt.title("Average Nodes Visited vs Trust Threshold α (Epinions)")
plt.legend()
plt.tight_layout()

# Save high-res
plt.savefig(os.path.join(output_dir, "epinions_avg_visited_vs_alpha.jpg"), dpi=600, format="jpg")
plt.savefig(os.path.join(output_dir, "epinions_avg_visited_vs_alpha.pdf"), dpi=600, format="pdf")
plt.close()

print(f"✅ Saved plots & table to: {output_dir}")
