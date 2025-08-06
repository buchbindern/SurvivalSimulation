import pandas as pd
import matplotlib.pyplot as plt 

def plot_results(results, model_type="rsf", **kwargs):
    
    # Concatenate DataFrames from the list of results
    data_mean = pd.concat(results[model_type]["mean"])
    data_std = pd.concat(results[model_type]["std"])
    
    # Create index based on censoring values
    index = pd.Index(data_mean["censoring"].round(3), name="mean percentage censoring")
    for df in (data_mean, data_std):
        df.drop("censoring", axis=1, inplace=True)
        df.index = index
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharex=True)
    
    # Concordance indexes plot
    cindex_columns = ["Actual C", "Harrel's C", "Uno's C"]
    data_mean_cindex = data_mean[cindex_columns]
    data_std_cindex = data_std[cindex_columns]
    
    data_mean_cindex.plot.bar(
        yerr=data_std_cindex,
        ax=axes[0],
        width=0.7,
        linewidth=0.5,
        capsize=4,
        **kwargs
    )
    axes[0].set_ylabel("Concordance")
    axes[0].set_title("Concordance Index Errors")
    axes[0].yaxis.grid(True, linestyle='--', alpha=0.7)
    axes[0].set_ylim(0, 1)  
    axes[0].axhline(0.0, color="gray", linewidth=0.8)
    
    # AUC comparison plot 
    auc_columns = ["Baseline AUC", "AUC"]
    data_mean_auc = data_mean[auc_columns]
    data_std_auc = data_std[auc_columns]
    
    # Plot the bar chart for AUC comparison
    data_mean_auc.plot.bar(
        yerr=data_std_auc,
        ax=axes[1],
        width=0.7,
        linewidth=0.5,
        capsize=4,
        **kwargs
    )
    axes[1].set_ylabel("AUC Score")
    axes[1].set_title("AUC Comparison: Baseline vs Model")
    axes[1].yaxis.grid(True, linestyle='--', alpha=0.7)
    axes[1].set_ylim(0, 1.0)
    
    # Brier score plot
    brier_column = ["Brier"]
    data_mean_brier = data_mean[brier_column]
    data_std_brier = data_std[brier_column]
    
    # Plot the bar chart for Brier scores
    data_mean_brier.plot.bar(
        yerr=data_std_brier,
        ax=axes[2],
        width=0.7,
        linewidth=0.5,
        capsize=4,
        **kwargs
    )
    axes[2].set_ylabel("Brier Score")
    axes[2].set_title("Integrated Brier Score")
    axes[2].yaxis.grid(True, linestyle='--', alpha=0.7)
    axes[2].set_ylim(0, 0.3)
    
    for ax in axes:
        ax.set_xlabel("Mean Percentage Censoring")
    
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='x', labelrotation=90)
    
    plt.tight_layout()
    return fig, axes