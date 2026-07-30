"""
Experiment 3: Information Theory
Dataset: Play Tennis Dataset
Task: Calculate the entropy and information gain for all input attributes and
      identify the attribute with the highest information gain (i.e. the best
      attribute to use as the root node of a Decision Tree, per ID3).
"""

import math
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1. Play Tennis Dataset (classic 14-instance benchmark dataset)
# ----------------------------------------------------------------------
data = {
    "Day":         [f"D{i}" for i in range(1, 15)],
    "Outlook":     ["Sunny", "Sunny", "Overcast", "Rain", "Rain", "Rain", "Overcast",
                     "Sunny", "Sunny", "Rain", "Sunny", "Overcast", "Overcast", "Rain"],
    "Temperature": ["Hot", "Hot", "Hot", "Mild", "Cool", "Cool", "Cool",
                     "Mild", "Cool", "Mild", "Mild", "Mild", "Hot", "Mild"],
    "Humidity":    ["High", "High", "High", "High", "Normal", "Normal", "Normal",
                     "High", "Normal", "Normal", "Normal", "High", "Normal", "High"],
    "Wind":        ["Weak", "Strong", "Weak", "Weak", "Weak", "Strong", "Strong",
                     "Weak", "Weak", "Weak", "Strong", "Strong", "Weak", "Strong"],
    "PlayTennis":  ["No", "No", "Yes", "Yes", "Yes", "No", "Yes",
                     "No", "Yes", "Yes", "Yes", "Yes", "Yes", "No"],
}
df = pd.DataFrame(data)

print("=" * 70)
print("EXPERIMENT 3: INFORMATION THEORY - Play Tennis Dataset")
print("=" * 70)
print(f"\n{df.to_string(index=False)}")

TARGET = "PlayTennis"
ATTRIBUTES = ["Outlook", "Temperature", "Humidity", "Wind"]


# ----------------------------------------------------------------------
# 2. Entropy function:  H(S) = - sum( p_i * log2(p_i) )
# ----------------------------------------------------------------------
def entropy(series):
    counts = series.value_counts()
    total = len(series)
    ent = 0.0
    for c in counts:
        p = c / total
        ent -= p * math.log2(p)
    return ent


# ----------------------------------------------------------------------
# 3. Overall (root) entropy of the target attribute
# ----------------------------------------------------------------------
target_counts = df[TARGET].value_counts()
total = len(df)
root_entropy = entropy(df[TARGET])

print("\n--- Step 1: Entropy of the Target Attribute (PlayTennis) ---")
for cls, cnt in target_counts.items():
    print(f"  {cls}: {cnt}/{total} = {cnt/total:.4f}")
print(f"\nH(PlayTennis) = - Σ p_i log2(p_i)")
terms = " - ".join([f"({cnt}/{total})log2({cnt}/{total})" for cls, cnt in target_counts.items()])
print(f"             = {root_entropy:.4f} bits")


# ----------------------------------------------------------------------
# 4. Information Gain for each attribute
#    IG(S, A) = H(S) - Σ (|Sv|/|S|) * H(Sv)
# ----------------------------------------------------------------------
def information_gain(df, attribute, target, base_entropy):
    total = len(df)
    weighted_entropy = 0.0
    breakdown = []
    for value, subset in df.groupby(attribute):
        p_v = len(subset) / total
        h_v = entropy(subset[target])
        weighted_entropy += p_v * h_v
        breakdown.append((value, len(subset), h_v, p_v))
    gain = base_entropy - weighted_entropy
    return gain, breakdown


print("\n" + "=" * 70)
print("Step 2: Information Gain for Each Attribute")
print("=" * 70)

ig_results = {}
for attr in ATTRIBUTES:
    gain, breakdown = information_gain(df, attr, TARGET, root_entropy)
    ig_results[attr] = gain
    print(f"\nAttribute: {attr}")
    for value, count, h_v, p_v in breakdown:
        yes = (df[(df[attr] == value) & (df[TARGET] == "Yes")]).shape[0]
        no = (df[(df[attr] == value) & (df[TARGET] == "No")]).shape[0]
        print(f"  {attr}={value:9s}  [Yes={yes}, No={no}]  count={count}  "
              f"H({value})={h_v:.4f}  weight={p_v:.4f}")
    weighted_sum = sum(p_v * h_v for _, _, h_v, p_v in breakdown)
    print(f"  Weighted Entropy Σ(|Sv|/|S|)H(Sv) = {weighted_sum:.4f}")
    print(f"  Information Gain IG(S,{attr}) = {root_entropy:.4f} - {weighted_sum:.4f} = {gain:.4f}")

# ----------------------------------------------------------------------
# 5. Identify attribute with highest information gain
# ----------------------------------------------------------------------
best_attr = max(ig_results, key=ig_results.get)
print("\n" + "=" * 70)
print("Step 3: Summary - Information Gain Ranking")
print("=" * 70)
sorted_ig = sorted(ig_results.items(), key=lambda x: x[1], reverse=True)
for attr, gain in sorted_ig:
    marker = "  <== HIGHEST (Best Root Node)" if attr == best_attr else ""
    print(f"  IG({attr:12s}) = {gain:.4f}{marker}")

print(f"\n==> ATTRIBUTE WITH HIGHEST INFORMATION GAIN: {best_attr} "
      f"(IG = {ig_results[best_attr]:.4f})")
print(f"    This attribute should be chosen as the ROOT NODE of the ID3 decision tree.")

# ----------------------------------------------------------------------
# 6. Save results
# ----------------------------------------------------------------------
with open("/home/claude/lab/results/exp3_results.txt", "w") as f:
    f.write("EXPERIMENT 3 RESULTS: INFORMATION THEORY (Play Tennis Dataset)\n\n")
    f.write(f"Root Entropy H(PlayTennis) = {root_entropy:.4f} bits\n\n")
    f.write("Information Gain per Attribute:\n")
    for attr, gain in sorted_ig:
        f.write(f"  IG({attr}) = {gain:.4f}\n")
    f.write(f"\nBest Attribute (Root Node): {best_attr} (IG = {ig_results[best_attr]:.4f})\n")

# ----------------------------------------------------------------------
# 7. Visualisation
# ----------------------------------------------------------------------
plt.figure(figsize=(8, 5))
attrs = [a for a, _ in sorted_ig]
gains = [g for _, g in sorted_ig]
colors = ["#5cb85c" if a == best_attr else "#337ab7" for a in attrs]
bars = plt.bar(attrs, gains, color=colors)
plt.ylabel("Information Gain (bits)")
plt.title("Information Gain of Each Attribute (Play Tennis Dataset)")
for bar, gain in zip(bars, gains):
    plt.text(bar.get_x() + bar.get_width() / 2, gain + 0.005, f"{gain:.4f}", ha="center")
plt.tight_layout()
plt.savefig("/home/claude/lab/results/exp3_information_gain.png", dpi=150)
plt.close()

print("\nResults and plot saved to /home/claude/lab/results/")
