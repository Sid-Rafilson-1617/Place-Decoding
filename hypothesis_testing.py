import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt
from scipy.stats import ranksums




# Directories for the saved CSV files
OB_csv_path = r"E:\place_decoding\data\bulb\OB_decoding_errors.csv"
OB_floorflip_csv_path = r"E:\place_decoding\data\bulb\floorflip_decoding_errors.csv"
hippocampus_csv_path = r"E:\place_decoding\data\hipp\hippocampus_decoding_errors.csv"


def plot_bars(OB_csv_path, hippocampus_csv_path):

    # Load the data
    ob_df = pd.read_csv(OB_csv_path)
    hippocampus_df = pd.read_csv(hippocampus_csv_path)

    # Convert string representations of lists back to lists
    ob_df["Decoding Error"] = ob_df["Decoding Error"].apply(ast.literal_eval)
    ob_df["Decoding Error Shuffled"] = ob_df["Decoding Error Shuffled"].apply(ast.literal_eval)
    hippocampus_df["Decoding Error"] = hippocampus_df["Decoding Error"].apply(ast.literal_eval)
    hippocampus_df["Decoding Error Shuffled"] = hippocampus_df["Decoding Error Shuffled"].apply(ast.literal_eval)

    # Collect individual data points for each category
    individual_points = {
        "OB": ob_df["Decoding Error"].apply(np.mean).values,
        "OB shuffled": ob_df["Decoding Error Shuffled"].apply(np.mean).values,
        "hippocampus": hippocampus_df["Decoding Error"].apply(np.mean).values,
        "hippocampus shuffled": hippocampus_df["Decoding Error Shuffled"].apply(np.mean).values
    }

    # Calculate bar heights for plotting
    means = {label: np.mean(points) for label, points in individual_points.items()}
    labels = list(means.keys())
    heights = list(means.values())

    # Perform rank-sum tests across each condition
    p_values = {}
    comparisons = [("OB", "OB shuffled"), ("hippocampus", "hippocampus shuffled"), ("OB", "hippocampus"), ("OB shuffled", "hippocampus shuffled")]
    for label1, label2 in comparisons:
        _, p = ranksums(individual_points[label1], individual_points[label2])
        p_values[(label1, label2)] = p

    # Significance level function
    def significance_label(p):
        if p < 0.001:
            return '***'
        elif p < 0.01:
            return '**'
        elif p < 0.05:
            return '*'
        else:
            return 'ns'

    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, heights, color=["blue", "orange", "blue", "orange"], alpha=0.7)
    ax.set_ylabel("Mean Decoding Error")
    ax.set_title("Decoding Error across Regions with Shuffled Controls")

    # Overlay individual points on each bar
    for i, label in enumerate(labels):
        y_values = individual_points[label]
        x_values = np.random.normal(bars[i].get_x() + bars[i].get_width() / 2, 0.05, size=len(y_values))  # Add jitter for visibility
        ax.scatter(x_values, y_values, color='black', s=10, alpha=0.8)

    # Add significance markers with staggered y-offsets for readability
    y_offset_base = max(heights) * 0.4  # Base space above bars for significance lines
    for i, ((label1, label2), p) in enumerate(p_values.items()):
        idx1, idx2 = labels.index(label1), labels.index(label2)
        x1, x2 = bars[idx1].get_x() + bars[idx1].get_width() / 2, bars[idx2].get_x() + bars[idx2].get_width() / 2
        y = max(bars[idx1].get_height(), bars[idx2].get_height()) + y_offset_base * (i + 1)
        
        # Draw the significance line and text label
        ax.plot([x1, x1, x2, x2], [y, y + y_offset_base * 0.2, y + y_offset_base * 0.2, y], color="black")
        ax.text((x1 + x2) / 2, y + y_offset_base * 0.25, significance_label(p), ha='center')

    plt.savefig(r"E:\place_decoding\decoding_error_comparison.png")
    plt.show()



def plot_scatter(OB_csv_path, hippocampus_csv_path):

    # Load the data
    ob_df = pd.read_csv(OB_csv_path)
    hippocampus_df = pd.read_csv(hippocampus_csv_path)

    # Prepare data for scatter plot
    sessions = []
    mean_errors = {
        "OB": {"real": [], "shuffled": [], "significant": []},
        "hippocampus": {"real": [], "shuffled": [], "significant": []}
    }

    # Analyze OB sessions
    for _, row in ob_df.iterrows():
        real_error = np.mean(row["Decoding Error"])
        shuffled_error = np.mean(row["Decoding Error Shuffled"])
        # Perform one-tailed rank-sum test (real < shuffled)
        stat, p_value = ranksums(row["Decoding Error"], row["Decoding Error Shuffled"], alternative="less")
        
        mean_errors["OB"]["real"].append(real_error)
        mean_errors["OB"]["shuffled"].append(shuffled_error)
        mean_errors["OB"]["significant"].append(p_value < 0.01)


    # Analyze hippocampus sessions
    for _, row in hippocampus_df.iterrows():
        real_error = np.mean(row["Decoding Error"])
        shuffled_error = np.mean(row["Decoding Error Shuffled"])
        # Perform one-tailed rank-sum test (real < shuffled)
        stat, p_value = ranksums(row["Decoding Error"], row["Decoding Error Shuffled"], alternative="less")
        
        mean_errors["hippocampus"]["real"].append(real_error)
        mean_errors["hippocampus"]["shuffled"].append(shuffled_error)
        mean_errors["hippocampus"]["significant"].append(p_value < 0.01)

    # Scatter plot
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"OB": "blue", "hippocampus": "green"}

    for region in ["OB", "hippocampus"]:
        x = mean_errors[region]["shuffled"]
        y = mean_errors[region]["real"]
        significance = mean_errors[region]["significant"]
        
        # Plot significant (filled) and non-significant (open) points
        ax.scatter(
            [x[i] for i in range(len(x)) if significance[i]], 
            [y[i] for i in range(len(y)) if significance[i]], 
            color=colors[region], edgecolor="black", label=f"{region} (significant)", marker="o", s=80
        )
        ax.scatter(
            [x[i] for i in range(len(x)) if not significance[i]], 
            [y[i] for i in range(len(y)) if not significance[i]], 
            facecolors="none", edgecolor=colors[region], label=f"{region} (not significant)", marker="o", s=80
        )

    # Add diagonal line
    ax.plot([0, 1000], [0, 1000], color="gray", linestyle="--")


    ax.set_xlabel("Mean Shuffled Decoding Error")
    ax.set_ylabel("Mean Real Decoding Error")
    ax.set_title("Mean Decoding Errors by Session (Real vs Shuffled)")
    ax.legend()
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1000)
    plt.savefig(r"E:\place_decoding\decoding_error_scatter.png")  # Save the scatter plot
    plt.show()  # Show the scatter plot


def plot_bars_floorflip(OB_floorflip_csv_path):
    # Load the data
    ob_floorflip_df = pd.read_csv(OB_floorflip_csv_path)

    # Convert string representations of lists back to lists
    ob_floorflip_df["DecodingError"] = ob_floorflip_df["DecodingError"].apply(ast.literal_eval)
    ob_floorflip_df["DecodingErrorShuffled"] = ob_floorflip_df["DecodingErrorShuffled"].apply(ast.literal_eval)

    # Get unique conditions from the "train/test" column
    conditions = ob_floorflip_df["split"].unique()

    # Collect individual data points for each condition
    individual_points = {
        condition: ob_floorflip_df[ob_floorflip_df["split"] == condition]["DecodingError"].apply(np.mean).values
        for condition in conditions
    }
    individual_points_shuffled = {
        f"{condition} shuffled": ob_floorflip_df[ob_floorflip_df["split"] == condition]["DecodingErrorShuffled"].apply(np.mean).values
        for condition in conditions
    }

    # Combine both normal and shuffled points for unified plotting and comparisons
    all_individual_points = {**individual_points, **individual_points_shuffled}

    # Calculate bar heights (means) for plotting
    means = {label: np.mean(points) for label, points in all_individual_points.items()}
    labels = []
    heights = []
    colors = []
    color_map = plt.cm.get_cmap("tab10", len(conditions))  # Use a distinct color for each condition
    
    # Build bars with alternating labels for real and shuffled
    for i, condition in enumerate(conditions):
        labels.extend([condition, f"{condition} shuffled"])
        heights.extend([means[condition], means[f"{condition} shuffled"]])
        colors.extend([color_map(i), color_map(i)])  # Same color pair for each condition and its shuffle

    # Perform rank-sum tests across each pair of conditions
    p_values = {}
    for condition in conditions:
        p_values[(condition, f"{condition} shuffled")] = ranksums(individual_points[condition], individual_points_shuffled[f"{condition} shuffled"])[1]

    # Significance level function
    def significance_label(p):
        if p < 0.001:
            return '***'
        elif p < 0.01:
            return '**'
        elif p < 0.05:
            return '*'
        else:
            return 'ns'

    # Create bar chart
    fig, ax = plt.subplots(figsize=(15, 10))
    bars = ax.bar(labels, heights, color=colors, alpha=0.7)
    ax.set_ylabel("Mean Decoding Error")
    ax.set_title("Decoding Error across train/test conditions with Shuffled Controls")

    # Overlay individual points on each bar
    for i, label in enumerate(labels):
        y_values = all_individual_points[label]
        x_values = np.random.normal(bars[i].get_x() + bars[i].get_width() / 2, 0.05, size=len(y_values))  # Add jitter for visibility
        ax.scatter(x_values, y_values, color='black', s=10, alpha=0.8)

    # Add significance markers with staggered y-offsets for readability
    y_offset_base = max(heights) * 0.1  # Base space above bars for significance lines
    for i, ((label1, label2), p) in enumerate(p_values.items()):
        idx1, idx2 = labels.index(label1), labels.index(label2)
        x1, x2 = bars[idx1].get_x() + bars[idx1].get_width() / 2, bars[idx2].get_x() + bars[idx2].get_width() / 2
        y = max(bars[idx1].get_height(), bars[idx2].get_height()) + y_offset_base * (i + 1)
        
        # Draw the significance line and text label
        ax.plot([x1, x1, x2, x2], [y, y + y_offset_base * 0.2, y + y_offset_base * 0.2, y], color="black")
        ax.text((x1 + x2) / 2, y + y_offset_base * 0.25, significance_label(p), ha='center')

    plt.savefig(r"E:\place_decoding\decoding_error_floorflip_comparison.png")
    plt.show()



def plot_bars_no_shuffle(OB_floorflip_csv_path):
    # Load the data
    ob_floorflip_df = pd.read_csv(OB_floorflip_csv_path)

    # Convert string representations of lists back to lists
    ob_floorflip_df["DecodingError"] = ob_floorflip_df["DecodingError"].apply(ast.literal_eval)

    # Get unique conditions from the "train/test" column
    conditions = ob_floorflip_df["split"].unique()

    # Collect individual data points for each condition (only real values, no shuffled)
    individual_points = {
        condition: ob_floorflip_df[ob_floorflip_df["split"] == condition]["DecodingError"].apply(np.mean).values
        for condition in conditions
    }

    # Calculate bar heights (means) for plotting
    means = {label: np.mean(points) for label, points in individual_points.items()}
    labels = list(means.keys())
    heights = list(means.values())

    # Perform rank-sum tests across all combinations of conditions
    p_values = {}
    for i, cond1 in enumerate(conditions):
        for cond2 in conditions[i + 1:]:
            p_value = ranksums(individual_points[cond1], individual_points[cond2])[1]
            p_values[(cond1, cond2)] = p_value

    # Significance level function
    def significance_label(p):
        if p < 0.001:
            return '***'
        elif p < 0.01:
            return '**'
        elif p < 0.05:
            return '*'
        else:
            return 'ns'

    # Create bar chart
    fig, ax = plt.subplots(figsize=(15, 10))
    color_map = plt.cm.get_cmap("tab10", len(conditions))  # Use a distinct color for each condition
    bars = ax.bar(labels, heights, color=[color_map(i) for i in range(len(conditions))], alpha=0.7)
    ax.set_ylabel("Mean Decoding Error")
    ax.set_title("Decoding Error across train/test Conditions")

    # Overlay individual points on each bar
    for i, label in enumerate(labels):
        y_values = individual_points[label]
        x_values = np.random.normal(bars[i].get_x() + bars[i].get_width() / 2, 0.05, size=len(y_values))  # Add jitter for visibility
        ax.scatter(x_values, y_values, color='black', s=10, alpha=0.8)

    # Display rank-sum test results as significance markers in the plot
    y_offset_base = max(heights) * 0.25  # Base space above bars for significance lines
    for i, ((cond1, cond2), p_value) in enumerate(p_values.items()):
        idx1, idx2 = labels.index(cond1), labels.index(cond2)
        x1, x2 = bars[idx1].get_x() + bars[idx1].get_width() / 2, bars[idx2].get_x() + bars[idx2].get_width() / 2
        y = max(bars[idx1].get_height(), bars[idx2].get_height()) + y_offset_base * (i + 1)
        
        # Draw the significance line and text label
        ax.plot([x1, x1, x2, x2], [y, y + y_offset_base * 0.2, y + y_offset_base * 0.2, y], color="black")
        ax.text((x1 + x2) / 2, y + y_offset_base * 0.25, significance_label(p_value), ha='center')

    plt.savefig(r"E:\place_decoding\decoding_error_floorflip_real_only.png")
    plt.show()


def plot_scatter_train_test(OB_floorflip_csv_path):
    # Load the data
    ob_floorflip_df = pd.read_csv(OB_floorflip_csv_path)

    # Convert string representations of lists back to lists
    ob_floorflip_df["DecodingError"] = ob_floorflip_df["DecodingError"].apply(ast.literal_eval)
    ob_floorflip_df["DecodingErrorShuffled"] = ob_floorflip_df["DecodingErrorShuffled"].apply(ast.literal_eval)

    # Prepare data for scatter plot
    conditions = ob_floorflip_df["split"].unique()
    mean_errors = {condition: {"real": [], "shuffled": [], "significant": []} for condition in conditions}

    # Analyze each train/test condition
    for condition in conditions:
        condition_data = ob_floorflip_df[ob_floorflip_df["split"] == condition]
        for _, row in condition_data.iterrows():
            real_error = np.mean(row["DecodingError"])
            shuffled_error = np.mean(row["DecodingErrorShuffled"])
            
            # Perform one-tailed rank-sum test (real < shuffled)
            stat, p_value = ranksums(row["DecodingError"], row["DecodingErrorShuffled"], alternative="less")
            significant = p_value < 0.01
            
            mean_errors[condition]["real"].append(real_error)
            mean_errors[condition]["shuffled"].append(shuffled_error)
            mean_errors[condition]["significant"].append(significant)

    # Scatter plot setup
    fig, ax = plt.subplots(figsize=(10, 8))
    color_map = plt.cm.get_cmap("tab10", len(conditions))

    for i, condition in enumerate(conditions):
        x = mean_errors[condition]["shuffled"]
        y = mean_errors[condition]["real"]
        significance = mean_errors[condition]["significant"]
        
        # Plot significant points (filled) and non-significant points (open)
        ax.scatter(
            [x[j] for j in range(len(x)) if significance[j]],
            [y[j] for j in range(len(y)) if significance[j]],
            color=color_map(i), edgecolor="black", label=f"{condition} (significant)", marker="o", s=80
        )
        ax.scatter(
            [x[j] for j in range(len(x)) if not significance[j]],
            [y[j] for j in range(len(y)) if not significance[j]],
            facecolors="none", edgecolor=color_map(i), label=f"{condition} (not significant)", marker="o", s=80
        )

    # Add a diagonal reference line for real vs shuffled error
    max_error = max([max(mean_errors[cond]["real"] + mean_errors[cond]["shuffled"]) for cond in conditions])
    ax.plot([0, max_error], [0, max_error], color="gray", linestyle="--")

    # Labels and title
    ax.set_xlabel("Mean Shuffled Decoding Error")
    ax.set_ylabel("Mean Real Decoding Error")
    ax.set_title("Mean Decoding Errors by Train/Test Condition (Real vs Shuffled)")
    ax.legend()
    plt.savefig(r"E:\place_decoding\decoding_error_scatter_train_test.png")  # Save the scatter plot
    plt.show()

    # print the fraction of signficant sessions in each condition
    for condition in conditions:
        fraction_significant = np.mean(mean_errors[condition]["significant"])
        print(f"Fraction of significant sessions for {condition}: {fraction_significant:.2f}")

if __name__ == "__main__":

    plot_bars_floorflip(OB_floorflip_csv_path)
    plot_bars_no_shuffle(OB_floorflip_csv_path)
    plot_scatter_train_test(OB_floorflip_csv_path)