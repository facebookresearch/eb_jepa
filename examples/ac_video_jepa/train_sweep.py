import argparse
import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml


def analyze_train_sweep_results(
    folder, enc_type="impala", last_k_epochs=None, average_over_seeds=False
):
    folder = Path(folder)
    all_results = []
    exp_results = {}
    for exp_folder in folder.iterdir():
        if not exp_folder.is_dir():
            continue
        plan_eval_folder = exp_folder / "plan_eval"
        if not plan_eval_folder.exists():
            continue
        success_rates = []
        for eval_folder in plan_eval_folder.iterdir():  # loop over epochs
            if eval_folder.is_dir():
                eval_csv = eval_folder / "eval.csv"
                if eval_csv.exists():
                    try:
                        df = pd.read_csv(eval_csv)
                        if "success_rate" in df.columns:
                            # Convert to numeric, handling any non-numeric values
                            success_rate = pd.to_numeric(
                                df["success_rate"].iloc[0], errors="coerce"
                            )
                            if not pd.isna(
                                success_rate
                            ):  # Only add if it's a valid number
                                success_rates.append(success_rate)
                    except Exception as e:
                        logging.warning(f"Error reading {eval_csv}: {e}")
        if success_rates:
            if last_k_epochs is not None:  # and len(success_rates) > last_k_epochs:
                success_rates = success_rates[-last_k_epochs:]
            exp_results[exp_folder.name] = success_rates

    for exp_folder_name, success_rates in exp_results.items():
        if not success_rates:
            continue
        params = parse_folder_name(exp_folder_name, enc_type=enc_type)
        if not params:
            continue
        mean_success_rate = sum(success_rates) / len(success_rates)
        result_entry = {
            **params,
            "mean_success_rate": mean_success_rate,
            "exp_folder": exp_folder_name,
        }
        all_results.append(result_entry)

    if not all_results:
        logging.warning("No valid results found")
        return None

    results_df = pd.DataFrame(all_results)

    # Save the raw results (before averaging over seeds)
    combined_path = folder / "combined_sweep_results_raw.csv"
    results_df.to_csv(combined_path, index=False)
    logging.info(f"Raw combined results saved to {combined_path}")

    # If seed averaging is requested, aggregate results by hyperparameters
    if average_over_seeds:
        # Identify parameter columns (exclude exp_folder and mean_success_rate)
        param_cols = [
            col
            for col in results_df.columns
            if col not in ["mean_success_rate", "exp_folder"]
        ]

        if param_cols:
            # Group by hyperparameters and aggregate
            aggregated = (
                results_df.groupby(param_cols, dropna=False)
                .agg(
                    {
                        "mean_success_rate": ["mean", "std", "count"],
                        "exp_folder": lambda x: ", ".join(
                            x
                        ),  # Keep track of which folders were averaged
                    }
                )
                .reset_index()
            )

            # Flatten the multi-level columns
            aggregated.columns = param_cols + [
                "mean_success_rate",
                "std_success_rate",
                "num_seeds",
                "exp_folders",
            ]

            # Use the aggregated results for further analysis
            results_df = aggregated
            logging.info(
                f"Averaged results over seeds. {len(results_df)} unique hyperparameter configurations."
            )
        else:
            logging.warning(
                "No parameter columns found for grouping. Skipping seed averaging."
            )

    # Save the final results (possibly averaged)
    final_path = folder / "combined_sweep_results.csv"
    results_df.to_csv(final_path, index=False)
    logging.info(f"Combined results saved to {final_path}")

    # Define metadata columns to exclude from hyperparameter analysis and plotting
    metadata_cols = [
        "mean_success_rate",
        "exp_folder",
        "std_success_rate",
        "num_seeds",
        "exp_folders",
    ]

    for param in results_df.columns:
        if param in metadata_cols:
            continue
        plt.figure(figsize=(10, 6))
        if results_df[param].dtype in ["object", "bool"]:
            param_means = (
                results_df.groupby(param)["mean_success_rate"]
                .agg(["mean", "std"])
                .reset_index()
            )
            plt.bar(
                param_means[param].astype(str),
                param_means["mean"],
                yerr=param_means["std"],
                capsize=5,
            )
        else:
            param_means = (
                results_df.groupby(param)["mean_success_rate"]
                .agg(["mean", "std"])
                .reset_index()
            )
            plt.errorbar(
                param_means[param],
                param_means["mean"],
                yerr=param_means["std"],
                marker="o",
                capsize=5,
            )
            plt.plot(param_means[param], param_means["mean"], "o-")
        plt.title(f"Effect of {param} on Mean Success Rate")
        plt.xlabel(param.replace("_", " ").title())
        plt.ylabel("Mean Success Rate")
        plt.grid(True, alpha=0.3)
        plot_path = folder / f"sweep_{param}_vs_success_rate.pdf"
        plt.savefig(plot_path, bbox_inches="tight", dpi=300)
        plt.close()
        logging.info(f"Plot saved to {plot_path}")

    # Filter out metadata columns from correlation analysis
    numeric_cols = [
        col
        for col in results_df.select_dtypes(include=["number"]).columns
        if col not in metadata_cols
    ] + ["mean_success_rate"]

    if len(numeric_cols) > 1:
        plt.figure(figsize=(10, 8))
        correlation_matrix = results_df[numeric_cols].corr()
        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap="RdBu_r",
            center=0,
            square=True,
            fmt=".2f",
        )
        plt.title("Hyperparameter and Success Rate Correlations")
        plt.tight_layout()
        heatmap_path = folder / "sweep_correlation_heatmap.pdf"
        plt.savefig(heatmap_path, bbox_inches="tight", dpi=300)
        plt.close()
        logging.info(f"Correlation heatmap saved to {heatmap_path}")

    # Filter out metadata columns from interaction plots
    numeric_params = [
        col
        for col in results_df.columns
        if results_df[col].dtype in ["int64", "float64"] and col not in metadata_cols
    ]
    for i, param1 in enumerate(numeric_params):
        for param2 in numeric_params[i + 1 :]:
            try:
                pivot_table = results_df.pivot_table(
                    values="mean_success_rate",
                    index=param1,
                    columns=param2,
                    aggfunc="mean",
                )
                if pivot_table.shape[0] > 1 and pivot_table.shape[1] > 1:
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(pivot_table, annot=True, cmap="viridis", fmt=".3f")
                    plt.gca().invert_yaxis()
                    plt.title(f"Interaction: {param1} vs {param2}")
                    plt.xlabel(param2.replace("_", " ").title())
                    plt.ylabel(param1.replace("_", " ").title())
                    interaction_path = folder / f"interaction_{param1}_{param2}.pdf"
                    plt.savefig(interaction_path, bbox_inches="tight", dpi=300)
                    plt.close()
                    logging.info(f"Interaction plot saved to {interaction_path}")
            except Exception as e:
                logging.warning(
                    f"Could not create interaction plot for {param1} vs {param2}: {e}"
                )

    best_idx = results_df["mean_success_rate"].idxmax()
    best_config = results_df.iloc[best_idx].to_dict()
    logging.info(f"Best configuration: {best_config}")
    logging.info(f"Best mean success rate: {best_config['mean_success_rate']:.4f}")

    propose_next_grid(results_df, folder)

    return best_config


def propose_next_grid(results_df, folder):
    """
    Analyze sweep results to find best configuration, study correlations and parameter interactions.

    Args:
        results_df: DataFrame with sweep results
        folder: Path to save the analysis results
    """
    optimal_values = {}

    # Define metadata columns to exclude from hyperparameter analysis
    metadata_cols = [
        "mean_success_rate",
        "exp_folder",
        "std_success_rate",
        "num_seeds",
        "exp_folders",
    ]

    # Get numeric parameters (excluding metadata columns)
    numeric_params = [
        col
        for col in results_df.columns
        if results_df[col].dtype in ["int64", "float64"] and col not in metadata_cols
    ]

    # Calculate correlation matrix for insights
    correlation_with_success = {}
    if len(numeric_params) > 0:
        for param in numeric_params:
            corr = results_df[[param, "mean_success_rate"]].corr().iloc[0, 1]
            correlation_with_success[param] = corr

    logging.info("\n" + "=" * 80)
    logging.info("CORRELATION ANALYSIS:")
    logging.info("=" * 80)
    for param, corr in sorted(
        correlation_with_success.items(), key=lambda x: abs(x[1]), reverse=True
    ):
        logging.info(f"  {param}: {corr:+.4f}")
    logging.info("=" * 80 + "\n")

    # Analyze parameter interactions
    param_interactions = {}
    if len(numeric_params) > 1:
        param_corr_matrix = results_df[numeric_params].corr()
        for i, param1 in enumerate(numeric_params):
            for param2 in numeric_params[i + 1 :]:
                corr = param_corr_matrix.loc[param1, param2]
                if abs(corr) > 0.5:  # Strong correlation threshold
                    param_interactions[(param1, param2)] = corr

    if param_interactions:
        logging.info("PARAMETER INTERACTIONS (|correlation| > 0.5):")
        logging.info("=" * 80)
        for (p1, p2), corr in sorted(
            param_interactions.items(), key=lambda x: abs(x[1]), reverse=True
        ):
            logging.info(f"  {p1} ↔ {p2}: {corr:+.4f}")
        logging.info("=" * 80 + "\n")

    # Find optimal values for each parameter
    for param in numeric_params:
        grouped = (
            results_df.groupby(param)["mean_success_rate"]
            .mean()
            .sort_values(ascending=False)
        )

        if len(grouped) < 2:
            logging.warning(f"Skipping {param}: insufficient data points")
            continue

        optimal_value = grouped.index[0]
        optimal_values[param] = optimal_value

    # Save optimal values configuration
    optimal_config = {
        "optimal_hyperparameters": optimal_values,
        "expected_success_rate": (
            float(
                results_df.loc[
                    (
                        results_df[list(optimal_values.keys())]
                        == pd.Series(optimal_values)
                    ).all(axis=1),
                    "mean_success_rate",
                ].max()
            )
            if len(optimal_values) > 0
            else None
        ),
    }

    optimal_path = folder / "optimal_configuration.yaml"
    with open(optimal_path, "w") as f:
        yaml.dump(optimal_config, f, default_flow_style=False)
    logging.info(f"Optimal configuration saved to {optimal_path}")

    # Save interaction-aware suggestions
    if param_interactions:
        interaction_suggestions = []
        for (p1, p2), corr in param_interactions.items():
            if p1 in optimal_values and p2 in optimal_values:
                interaction_suggestions.append(
                    {
                        "param1": p1,
                        "param2": p2,
                        "correlation": corr,
                        "optimal_value_1": optimal_values[p1],
                        "optimal_value_2": optimal_values[p2],
                        "suggestion": (
                            "Consider sweeping these together"
                            if abs(corr) > 0.7
                            else "Moderate interaction"
                        ),
                    }
                )

        if interaction_suggestions:
            interaction_df = pd.DataFrame(interaction_suggestions)
            interaction_path = folder / "parameter_interactions.csv"
            interaction_df.to_csv(interaction_path, index=False)
            logging.info(
                f"Parameter interaction suggestions saved to {interaction_path}"
            )

    # Log comprehensive summary
    logging.info("\n" + "=" * 80)
    logging.info("OPTIMAL HYPERPARAMETER CONFIGURATION:")
    logging.info("=" * 80)
    for param, value in optimal_values.items():
        corr = correlation_with_success.get(param, 0)
        logging.info(f"  {param}: {value} (correlation with success: {corr:+.4f})")
    logging.info("=" * 80 + "\n")

    # Provide sweep strategy recommendations
    logging.info("=" * 80)
    logging.info("SWEEP STRATEGY RECOMMENDATIONS:")
    logging.info("=" * 80)

    high_impact = [p for p, c in correlation_with_success.items() if abs(c) > 0.3]
    low_impact = [p for p, c in correlation_with_success.items() if abs(c) < 0.15]

    if high_impact:
        logging.info(f"  HIGH IMPACT parameters (prioritize): {', '.join(high_impact)}")
    if low_impact:
        logging.info(
            f"  LOW IMPACT parameters (consider fixing): {', '.join(low_impact)}"
        )

    if param_interactions:
        strongly_coupled = [
            (p1, p2) for (p1, p2), c in param_interactions.items() if abs(c) > 0.7
        ]
        if strongly_coupled:
            logging.info(f"  STRONGLY COUPLED parameters (sweep together):")
            for p1, p2 in strongly_coupled:
                logging.info(f"    - {p1} ↔ {p2}")

    logging.info("=" * 80 + "\n")


def parse_folder_name(folder_name, enc_type="impala"):
    folder_name_clean = re.sub(r"_seed\d+$", "", folder_name)
    # ################ HARDCODE THE PATTERN #############
    # pattern = rf"{enc_type}_encskTrue_prskTrue_cov8_std16_simt12_idm1_spFalse_useprojFalse_idmprojFalse_simtprojFalse_1sttFalse_rolllast_dstc32_henc32_hpre32"
    # pattern = rf"{enc_type}_encskTrue_prskTrue_cov0.0_std16_simt12_idm1_spFalse_useprojFalse_idmprojFalse_simtprojFalse_1sttFalse_rolllast_dstc32_henc32_hpre32_lr([0-9\.e\-]+)_wd([0-9\.e\-]+)"
    # pattern = rf"{enc_type}_encskFalse_prskFalse_cov8_std8_simt16_idm4_spFalse_useprojFalse_idmprojFalse_simtprojFalse_1sttFalse_rollall_dstc8_henc32_hpre32_lr([0-9\.e\-]+)_wd([0-9\.e\-]+)"
    # pattern = rf"{enc_type}_encskFalse_prskFalse_cov8_std8_simt16_idm4_spFalse_useprojFalse_idmprojFalse_simtprojFalse_1sttFalse_rolllast_dstc8_henc32_hpre32_lr([0-9\.e\-]+)_wd([0-9\.e\-]+)"
    pattern = rf"{enc_type}_encskTrue_prskTrue_cov(\d+(?:\.\d+)?)_std(\d+(?:\.\d+)?)_simt(\d+(?:\.\d+)?)_idm(\d+(?:\.\d+)?)"
    # pattern = rf"{enc_type}_encskFalse_prskFalse_cov(\d+(?:\.\d+)?)_std(\d+(?:\.\d+)?)_simt(\d+(?:\.\d+)?)_idm(\d+(?:\.\d+)?)"
    match = re.search(pattern, folder_name_clean)
    if match:
        params = {
            "cov_coeff": float(match.group(1)),
            "std_coeff": float(match.group(2)),
            "sim_coeff_t": float(match.group(3)),
            "idm_coeff": float(match.group(4)),
        }
        # params = {
        #     "lr": float(match.group(1)),
        #     # "lr": float(match.group(2)),
        #     "weight_decay": float(match.group(2)),
        #     # "weight_decay": float(match.group(3)),
        # }
        return params
    # #####################################################
    return {}


def main(
    sweep_folder=None, enc_type="impala", last_k_epochs=None, average_over_seeds=False
):
    if sweep_folder is None:
        sweep_folder = Path("/checkpoint/amaia/video/basileterv/experiment/eb_jepa")
        if not sweep_folder.exists():
            sweep_folder = Path("logs")
    else:
        sweep_folder = Path(sweep_folder)
    if not sweep_folder.exists():
        logging.error(f"Sweep folder {sweep_folder} does not exist")
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    seed_averaging_msg = " with seed averaging enabled" if average_over_seeds else ""
    logging.info(
        f"Analyzing training sweep results in {sweep_folder} using last {last_k_epochs} epochs{seed_averaging_msg}"
    )
    best_config = analyze_train_sweep_results(
        folder=sweep_folder,
        enc_type=enc_type,
        last_k_epochs=last_k_epochs,
        average_over_seeds=average_over_seeds,
    )
    if best_config:
        logging.info("Analysis complete.")
        logging.info(
            f"Best configuration found with success rate: {best_config['mean_success_rate']:.4f}"
        )
    else:
        logging.info("Analysis complete. No valid results found.")


if __name__ == "__main__":
    """
    Example command :
        python examples/ac_video_jepa/train_sweep.py --sweep-folder /checkpoint/amaia/video/basileterv/experiment/eb_jepa/randwall_imp_sweep
        python examples/ac_video_jepa/train_sweep.py --sweep-folder /checkpoint/amaia/video/basileterv/experiment/eb_jepa/randwall_imp_sweep --enc-type conv
        python examples/ac_video_jepa/train_sweep.py --sweep-folder /checkpoint/amaia/video/basileterv/experiment/eb_jepa/rand_conv3d_bs128_sweep/ --enc-type conv3d
        python examples/ac_video_jepa/train_sweep.py --sweep-folder /checkpoint/amaia/video/basileterv/experiment/eb_jepa/randwall_imp_fixeval_sweep
        python examples/ac_video_jepa/train_sweep.py --sweep-folder /checkpoint/amaia/video/basileterv/experiment/eb_jepa/randwall_imp_fixeval_opt_sweep
        python examples/ac_video_jepa/train_sweep.py --sweep-folder /checkpoint/amaia/video/basileterv/experiment/eb_jepa/rand_conv3d_bs128_opt_sweep --enc-type conv3d
        python examples/ac_video_jepa/train_sweep.py --sweep-folder /checkpoint/amaia/video/basileterv/experiment/eb_jepa/rand_imp_bs384AdamW_losssweep --average-over-seeds
    """
    parser = argparse.ArgumentParser(
        description="Analyze training sweep results from run_exp.sh"
    )
    parser.add_argument(
        "--sweep-folder",
        type=str,
        help="Path to the folder containing sweep results",
        default=None,
    )
    parser.add_argument(
        "--enc-type",
        type=str,
        help="Type of encoder used in experiments (e.g., 'impala')",
        default="impala",
    )
    parser.add_argument(
        "--last_k_epochs",
        type=int,
        help="Number of last epochs to consider for averaging success rate",
        default=3,
    )
    parser.add_argument(
        "--average-over-seeds",
        action="store_true",
        help="Average success rates over different seeds with the same hyperparameters",
    )
    args = parser.parse_args()
    main(
        sweep_folder=args.sweep_folder,
        enc_type=args.enc_type,
        last_k_epochs=args.last_k_epochs,
        average_over_seeds=args.average_over_seeds,
    )
