"""
Experiment 1: Probability Theory
Dataset: Breast Cancer Wisconsin (Diagnostic) Dataset
Task: Calculate the probability of each class (Malignant / Benign) and use the
      computed probabilities (via a Gaussian likelihood model) to predict the
      class of a new data instance.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1. Load Dataset
# ----------------------------------------------------------------------
data = load_breast_cancer(as_frame=True)
df = data.frame.copy()
# sklearn encodes target as 0 = malignant, 1 = benign
df["diagnosis"] = df["target"].map({0: "Malignant", 1: "Benign"})

print("=" * 70)
print("EXPERIMENT 1: PROBABILITY THEORY - Breast Cancer Wisconsin Dataset")
print("=" * 70)
print(f"\nTotal instances: {len(df)}")
print(f"Total features used (for demo, first 4): {list(data.feature_names[:4])}")
print(df[["mean radius", "mean texture", "mean perimeter", "mean area", "diagnosis"]].head())

# ----------------------------------------------------------------------
# 2. Prior Probability of each class  P(Malignant), P(Benign)
# ----------------------------------------------------------------------
class_counts = df["diagnosis"].value_counts()
n_total = len(df)
prior = (class_counts / n_total).to_dict()

print("\n--- Step 1: Prior (Class) Probabilities ---")
for cls, cnt in class_counts.items():
    print(f"P({cls}) = {cnt}/{n_total} = {prior[cls]:.4f}")

# ----------------------------------------------------------------------
# 3. Likelihood model: P(feature | class) modelled as Gaussian
#    We use a subset of 4 features for a clear, interpretable demo
# ----------------------------------------------------------------------
features = ["mean radius", "mean texture", "mean perimeter", "mean area"]

stats = {}
for cls in ["Malignant", "Benign"]:
    subset = df[df["diagnosis"] == cls][features]
    stats[cls] = {
        "mean": subset.mean(),
        "std": subset.std(),
    }

print("\n--- Step 2: Class-conditional Mean & Std. Dev (Gaussian likelihood params) ---")
for cls in stats:
    print(f"\n{cls}:")
    summary = pd.DataFrame({"mean": stats[cls]["mean"], "std": stats[cls]["std"]})
    print(summary)


def gaussian_pdf(x, mean, std):
    """Gaussian probability density function."""
    exponent = np.exp(-((x - mean) ** 2) / (2 * std ** 2))
    return (1.0 / (np.sqrt(2 * np.pi) * std)) * exponent


def predict_class(instance, features, stats, prior):
    """
    Naive Bayes style prediction:
    Posterior(class) proportional to  Prior(class) * Product_i P(feature_i | class)
    """
    posteriors_raw = {}
    likelihood_breakdown = {}

    for cls in stats:
        likelihood = 1.0
        per_feature = {}
        for f in features:
            p = gaussian_pdf(instance[f], stats[cls]["mean"][f], stats[cls]["std"][f])
            per_feature[f] = p
            likelihood *= p
        likelihood_breakdown[cls] = per_feature
        posteriors_raw[cls] = prior[cls] * likelihood

    # Normalise so the two posteriors sum to 1
    total = sum(posteriors_raw.values())
    posteriors = {cls: val / total for cls, val in posteriors_raw.items()}
    predicted = max(posteriors, key=posteriors.get)
    return predicted, posteriors, likelihood_breakdown


# ----------------------------------------------------------------------
# 4. Predict the class of a NEW data instance
# ----------------------------------------------------------------------
# New / unseen instance (values are realistic and taken close to the
# malignant-class region of the feature space)
new_instance = {
    "mean radius": 17.5,
    "mean texture": 21.0,
    "mean perimeter": 115.0,
    "mean area": 950.0,
}

print("\n--- Step 3: New Instance to Classify ---")
print(new_instance)

predicted_class, posteriors, likelihoods = predict_class(new_instance, features, stats, prior)

print("\n--- Step 4: Likelihoods P(feature | class) for the new instance ---")
for cls in likelihoods:
    print(f"\n{cls}:")
    for f, p in likelihoods[cls].items():
        print(f"   P({f} = {new_instance[f]} | {cls}) = {p:.6e}")

print("\n--- Step 5: Posterior Probabilities (normalised) ---")
for cls, p in posteriors.items():
    print(f"P({cls} | data) = {p:.6f}")

print(f"\n==> PREDICTED CLASS: {predicted_class}")

# ----------------------------------------------------------------------
# 5. A second test instance (closer to benign profile) for comparison
# ----------------------------------------------------------------------
new_instance_2 = {
    "mean radius": 9.5,
    "mean texture": 14.0,
    "mean perimeter": 60.0,
    "mean area": 280.0,
}
predicted_class_2, posteriors_2, _ = predict_class(new_instance_2, features, stats, prior)
print("\n--- Additional Test Instance (Benign-like profile) ---")
print(new_instance_2)
for cls, p in posteriors_2.items():
    print(f"P({cls} | data) = {p:.6f}")
print(f"==> PREDICTED CLASS: {predicted_class_2}")

# ----------------------------------------------------------------------
# 6. Save results to file
# ----------------------------------------------------------------------
results_txt = []
results_txt.append("EXPERIMENT 1 RESULTS: PROBABILITY THEORY\n")
results_txt.append(f"Total Instances: {n_total}\n")
for cls, cnt in class_counts.items():
    results_txt.append(f"P({cls}) = {cnt}/{n_total} = {prior[cls]:.4f}\n")
results_txt.append(f"\nNew Instance 1: {new_instance}\n")
for cls, p in posteriors.items():
    results_txt.append(f"  P({cls} | data) = {p:.6f}\n")
results_txt.append(f"  Predicted Class: {predicted_class}\n")
results_txt.append(f"\nNew Instance 2: {new_instance_2}\n")
for cls, p in posteriors_2.items():
    results_txt.append(f"  P({cls} | data) = {p:.6f}\n")
results_txt.append(f"  Predicted Class: {predicted_class_2}\n")

with open("/home/claude/lab/results/exp1_results.txt", "w") as f:
    f.writelines(results_txt)

# ----------------------------------------------------------------------
# 7. Visualisations
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Bar chart: prior probabilities
axes[0].bar(class_counts.index, [prior[c] for c in class_counts.index],
            color=["#d9534f", "#5cb85c"])
axes[0].set_title("Prior Class Probabilities")
axes[0].set_ylabel("Probability")
for i, c in enumerate(class_counts.index):
    axes[0].text(i, prior[c] + 0.01, f"{prior[c]:.3f}", ha="center")

# Bar chart: posterior probabilities for new instance 1
axes[1].bar(posteriors.keys(), posteriors.values(), color=["#d9534f", "#5cb85c"])
axes[1].set_title("Posterior Probabilities for New Instance")
axes[1].set_ylabel("Probability")
for i, (c, p) in enumerate(posteriors.items()):
    axes[1].text(i, p + 0.01, f"{p:.3f}", ha="center")

plt.tight_layout()
plt.savefig("/home/claude/lab/results/exp1_probabilities.png", dpi=150)
plt.close()

# Feature distribution plot (mean radius) by class
plt.figure(figsize=(7, 5))
for cls, color in zip(["Malignant", "Benign"], ["#d9534f", "#5cb85c"]):
    subset = df[df["diagnosis"] == cls]["mean radius"]
    plt.hist(subset, bins=25, alpha=0.6, label=cls, color=color, density=True)
plt.axvline(new_instance["mean radius"], color="black", linestyle="--",
            label=f'New instance (radius={new_instance["mean radius"]})')
plt.xlabel("Mean Radius")
plt.ylabel("Density")
plt.title("Distribution of Mean Radius by Class")
plt.legend()
plt.tight_layout()
plt.savefig("/home/claude/lab/results/exp1_feature_distribution.png", dpi=150)
plt.close()

print("\nResults and plots saved to /home/claude/lab/results/")
