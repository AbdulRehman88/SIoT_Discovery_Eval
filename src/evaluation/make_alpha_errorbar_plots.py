# Python 3.8/3.9 compatible

import argparse
import math
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import yaml

# ----------------------------
# Utilities
# ----------------------------

def load_config(cfg_path: str) -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def binomial_std(p: float, n: int) -> float:
    return math.sqrt(max(p * (1 - p), 0.0) / max(n, 1))

# ----------------------------
# Data loading
# ----------------------------

def load_mode_csv(base_dir: Path, filename: str) -> pd.DataFrame:
    fp = base_dir / "results" / filename
    if not fp.exists():
        raise FileNotFoundError(f"Missing file: {fp}")
    df = pd.read_csv(fp)
    expected = {"Dataset","Alpha","LCCRatio","SuccessRate","AvgHops","AvgTime","AvgVisited","Trials"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"{fp.name} is missing columns: {missing}")
    return df

def load_optional_pertrial(base_dir: Path, pattern: str) -> Optional[pd.DataFrame]:
    """
    Optional per-trial CSV with columns:
    Dataset, Mode, Alpha, Trial, Success(0/1), Hops, Time, Visited
    """
    fp = base_dir / "results" / pattern
    return pd.read_csv(fp) if fp.exists() else None

# ----------------------------
# Preparation
# ----------------------------

def attach_successrate_errors(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["SR_Std"] = out.apply(
        lambda r: binomial_std(float(r["SuccessRate"]), int(r["Trials"])), axis=1
    )
    return out

def pertrial_std_by_metric(pertrial: pd.DataFrame, mode_name: str) -> Optional[pd.DataFrame]:
    if pertrial is None or pertrial.empty:
        return None
    needed = {"Dataset","Mode","Alpha","Hops","Time","Visited"}
    if not needed.issubset(pertrial.columns):
        return None
    sub = pertrial[pertrial["Mode"].str.lower() == mode_name.lower()].copy()
    if sub.empty:
        return None
    g = sub.groupby(["Dataset","Alpha"], as_index=False).agg(
        AH_Std=("Hops","std"),
        AT_Std=("Time","std"),
        AV_Std=("Visited","std"),
    )
    return g

# ----------------------------
# Plotting (combined Greedy & Greedy+Fallback)
# ----------------------------

def combined_plot(
    greedy_df: pd.DataFrame,
    fallback_df: pd.DataFrame,
    out_pdf: Path,
    metric_col: str,
    y_label: str,
    title: str,
    greedy_std_map: Optional[pd.DataFrame] = None,
    fallback_std_map: Optional[pd.DataFrame] = None,
    dpi: int = 600
):
    ensure_dir(out_pdf.parent)

    # Prepare legend labels like "Dataset (Greedy)" and "Dataset (Fallback)"
    datasets = sorted(set(greedy_df["Dataset"].unique()) | set(fallback_df["Dataset"].unique()))

    plt.figure(figsize=(9, 5.5))

    for ds in datasets:
        gds = greedy_df[greedy_df["Dataset"] == ds].sort_values("Alpha")
        fds = fallback_df[fallback_df["Dataset"] == ds].sort_values("Alpha")

        # X & Y
        xg, yg = gds["Alpha"].values, gds[metric_col].values
        xf, yf = fds["Alpha"].values, fds[metric_col].values

        # Determine error bars
        if metric_col == "SuccessRate":
            yerr_g = gds["SR_Std"].values if "SR_Std" in gds.columns else None
            yerr_f = fds["SR_Std"].values if "SR_Std" in fds.columns else None
        else:
            yerr_g = None
            yerr_f = None
            # attach per-trial std if provided
            if greedy_std_map is not None:
                mg = gds.merge(greedy_std_map, on=["Dataset","Alpha"], how="left")
                std_col = {"AvgHops":"AH_Std","AvgTime":"AT_Std","AvgVisited":"AV_Std"}.get(metric_col)
                if std_col and std_col in mg.columns and mg[std_col].notna().any():
                    yerr_g = mg[std_col].values
            if fallback_std_map is not None:
                mf = fds.merge(fallback_std_map, on=["Dataset","Alpha"], how="left")
                std_col = {"AvgHops":"AH_Std","AvgTime":"AT_Std","AvgVisited":"AV_Std"}.get(metric_col)
                if std_col and std_col in mf.columns and mf[std_col].notna().any():
                    yerr_f = mf[std_col].values

        # Plot Greedy (solid) and Fallback (dashed)
        lbl_g = f"{ds} (Greedy)"
        lbl_f = f"{ds} (Fallback)"
        if yerr_g is not None:
            plt.errorbar(xg, yg, yerr=yerr_g, fmt="-o", capsize=3, label=lbl_g)
        else:
            plt.plot(xg, yg, "-o", label=lbl_g)

        if yerr_f is not None:
            plt.errorbar(xf, yf, yerr=yerr_f, fmt="--s", capsize=3, label=lbl_f)
        else:
            plt.plot(xf, yf, "--s", label=lbl_f)

    plt.xlabel("Trust Threshold (α)")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_pdf, dpi=dpi)
    plt.close()

def epinions_avgvisited_combined(
    greedy_df: pd.DataFrame,
    fallback_df: pd.DataFrame,
    out_pdf: Path,
    dpi: int = 600,
    greedy_std_map: Optional[pd.DataFrame] = None,
    fallback_std_map: Optional[pd.DataFrame] = None,
):
    ds = "Epinions"
    gds = greedy_df[greedy_df["Dataset"] == ds].sort_values("Alpha")
    fds = fallback_df[fallback_df["Dataset"] == ds].sort_values("Alpha")

    xg, yg = gds["Alpha"].values, gds["AvgVisited"].values
    xf, yf = fds["Alpha"].values, fds["AvgVisited"].values

    yerr_g = yerr_f = None
    if greedy_std_map is not None:
        mg = gds.merge(greedy_std_map, on=["Dataset","Alpha"], how="left")
        if "AV_Std" in mg.columns and mg["AV_Std"].notna().any():
            yerr_g = mg["AV_Std"].values
    if fallback_std_map is not None:
        mf = fds.merge(fallback_std_map, on=["Dataset","Alpha"], how="left")
        if "AV_Std" in mf.columns and mf["AV_Std"].notna().any():
            yerr_f = mf["AV_Std"].values

    plt.figure(figsize=(8, 5))
    if yerr_g is not None:
        plt.errorbar(xg, yg, yerr=yerr_g, fmt="-o", capsize=3, label="Epinions (Greedy)")
    else:
        plt.plot(xg, yg, "-o", label="Epinions (Greedy)")

    if yerr_f is not None:
        plt.errorbar(xf, yf, yerr=yerr_f, fmt="--s", capsize=3, label="Epinions (Fallback)")
    else:
        plt.plot(xf, yf, "--s", label="Epinions (Fallback)")

    plt.xlabel("Trust Threshold (α)")
    plt.ylabel("Average Nodes Visited")
    plt.title("Average Nodes Visited vs Trust Threshold  (Epinions)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    ensure_dir(out_pdf.parent)
    plt.savefig(out_pdf, dpi=dpi)
    plt.close()

# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate combined α–sensitivity plots with error bars.")
    default_cfg = (Path(__file__).resolve().parent.parent.parent / "config.yaml")
    parser.add_argument("--config", default=str(default_cfg), help="Path to config.yaml")
    parser.add_argument("--greedy_csv", default="trust_threshold_sensitivity_top6_trials100_greedy.csv")
    parser.add_argument("--fallback_csv", default="trust_threshold_sensitivity_top6_trials100_fallback.csv")
    parser.add_argument("--pertrial_csv", default="trust_threshold_sensitivity_top6_trials100_pertrial.csv")
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()

    cfg = load_config(args.config)
    base_dir = Path(cfg["base_dir"])
    results_dir = base_dir / "results"
    out_dir = results_dir / "plots_with_errorbars"
    ensure_dir(out_dir)

    greedy_df = load_mode_csv(base_dir, args.greedy_csv)
    fallback_df = load_mode_csv(base_dir, args.fallback_csv)

    # Success Rate error bars for both modes
    greedy_df = attach_successrate_errors(greedy_df)
    fallback_df = attach_successrate_errors(fallback_df)

    # Optional per-trial std maps (for hops/time/visited)
    pertrial_df = load_optional_pertrial(base_dir, args.pertrial_csv)
    greedy_std_map = pertrial_std_by_metric(pertrial_df, "greedy")
    fallback_std_map = pertrial_std_by_metric(pertrial_df, "fallback")

    # Combined plots (Greedy vs Greedy+Fallback)
    combined_plot(
        greedy_df, fallback_df,
        out_pdf=out_dir / "combined_success_rate_vs_alpha.pdf",
        metric_col="SuccessRate", y_label="Success Rate",
        title="Success Rate vs Trust Threshold",
        greedy_std_map=greedy_std_map, fallback_std_map=fallback_std_map,
        dpi=args.dpi
    )

    combined_plot(
        greedy_df, fallback_df,
        out_pdf=out_dir / "combined_avg_hops_vs_alpha.pdf",
        metric_col="AvgHops", y_label="Average Hops",
        title="Average Hops vs Trust Threshold",
        greedy_std_map=greedy_std_map, fallback_std_map=fallback_std_map,
        dpi=args.dpi
    )

    combined_plot(
        greedy_df, fallback_df,
        out_pdf=out_dir / "combined_avg_time_vs_alpha.pdf",
        metric_col="AvgTime", y_label="Average Time (s)",
        title="Average Time (s) vs Trust Threshold",
        greedy_std_map=greedy_std_map, fallback_std_map=fallback_std_map,
        dpi=args.dpi
    )

    combined_plot(
        greedy_df, fallback_df,
        out_pdf=out_dir / "combined_avg_visited_vs_alpha.pdf",
        metric_col="AvgVisited", y_label="Average Nodes Visited",
        title="Average Nodes Visited vs Trust Threshold",
        greedy_std_map=greedy_std_map, fallback_std_map=fallback_std_map,
        dpi=args.dpi
    )

    combined_plot(
        greedy_df, fallback_df,
        out_pdf=out_dir / "combined_lcc_ratio_vs_alpha.pdf",
        metric_col="LCCRatio", y_label="Largest Connected Component Ratio",
        title="Largest Connected Component Ratio vs Trust Threshold",
        greedy_std_map=greedy_std_map, fallback_std_map=fallback_std_map,
        dpi=args.dpi
    )

    # Epinions-only combined AvgVisited (to mirror your existing single-dataset panel)
    epinions_avgvisited_combined(
        greedy_df, fallback_df,
        out_pdf=out_dir / "epinions_avg_visited_vs_alpha.pdf",
        dpi=args.dpi,
        greedy_std_map=greedy_std_map, fallback_std_map=fallback_std_map
    )

    print(f"Saved combined plots to: {out_dir}")

if __name__ == "__main__":
    main()
