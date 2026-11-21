import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Load dataset
df = pd.read_csv("customers.csv")

# 2. Select demographic & behavioral features and scale them
features = ["Age", "AnnualIncome_INR", "PurchaseFrequency", "AvgOrderValue_INR"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# 3. Elbow method to check optimal clusters (saved as image)
inertia = []
for k in range(1, 7):
    km = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.figure(figsize=(6, 4))
plt.plot(range(1, 7), inertia, marker="o")
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.tight_layout()
plt.savefig("elbow_curve.png")
plt.close()

# 4. Fit K-Means with 3 segments (adjust K based on your dataset size)
kmeans = KMeans(n_clusters=3, init="k-means++", random_state=42, n_init=10)
df["Segment"] = kmeans.fit_predict(X_scaled)

# 5. Print segment summary & top preferences
print("\n--- Segment Averages ---")
print(df.groupby("Segment")[features].mean().round(1))

print("\n--- Top Category & Region per Segment ---")
for seg, grp in df.groupby("Segment"):
    top_cat = grp["PreferredCategory"].mode()[0]
    top_reg = grp["Region"].mode()[0]
    print(f"Segment {seg} (n={len(grp)}): Top Category = {top_cat}, Top Region = {top_reg}")

# 6. Scatter plot of Income vs Avg Order Value by Segment
plt.figure(figsize=(7, 5))
for seg, grp in df.groupby("Segment"):
    plt.scatter(grp["AnnualIncome_INR"], grp["AvgOrderValue_INR"], label=f"Segment {seg}", s=80)

plt.title("Customer Segments: Annual Income vs Avg Order Value")
plt.xlabel("Annual Income (INR)")
plt.ylabel("Avg Order Value (INR)")
plt.legend()
plt.tight_layout()
plt.savefig("customer_segments.png")
plt.close()

# 7. Save segmented output for reporting or Power BI
df.to_csv("segmented_customers.csv", index=False)
print("\nSaved segmented_customers.csv, elbow_curve.png, and customer_segments.png")