import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

features = [
    "age",
    "visits_per_month",
    "avg_visit_minutes",
    "classes_per_month",
    "months_member",
    "monthly_extras_spend",
]

training = pd.read_csv("training_set.csv")
test = pd.read_csv("test_set.csv")

X_train = training[features]
X_test = test[features]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

ks = list(range(1, 9))
inertias = []
test_spreads = []
final_k = 4
final_model = None
train_labels = None
test_labels = None

for k in ks:
    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    model.fit(X_train_scaled)

    inertia = round(model.inertia_, 1)
    inertias.append(inertia)

    labels = model.predict(X_test_scaled)
    centers = model.cluster_centers_[labels]
    test_spread = round(float(np.sum((X_test_scaled - centers) ** 2)), 1)
    test_spreads.append(test_spread)

    print(f"k={k} training inertia={inertia} test spread={test_spread}")

    if k == final_k:
        final_model = model
        train_labels = model.labels_
        test_labels = labels

plt.plot(ks, inertias, marker="o")
plt.xlabel("k")
plt.ylabel("inertia")
plt.title("Training elbow")
plt.xticks(ks)
plt.tight_layout()
plt.savefig("training_elbow.png")
plt.close()

plt.plot(ks, inertias, marker="o", label="Training inertia")
plt.plot(ks, test_spreads, marker="o", label="Test spread")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Sum of squared distances")
plt.title("Training and Test Spread by Number of Clusters")
plt.xticks(ks)
plt.legend()
plt.tight_layout()
plt.savefig("elbow_train_test.png")
plt.close()

print()
print(f"Training members in each cluster (k={final_k})")
for cluster in range(final_k):
    count = int(np.sum(train_labels == cluster))
    print(f"cluster {cluster}: {count}")

centers_original = scaler.inverse_transform(final_model.cluster_centers_)
centers_table = pd.DataFrame(centers_original, columns=features).round(1)
centers_table.insert(0, "cluster", range(final_k))

print()
print(centers_table.to_string(index=False))

print()
print(f"Total members in each cluster (training + test, k={final_k})")
for cluster in range(final_k):
    total = int(np.sum(train_labels == cluster) + np.sum(test_labels == cluster))
    print(f"cluster {cluster}: {total}")

segments = {
    0: {"segment_name": "New Light Users", "gets_credit": True},
    1: {"segment_name": "Long-Term Regulars", "gets_credit": True},
    2: {"segment_name": "Frequent Gym Users", "gets_credit": False},
    3: {"segment_name": "High-Value Class Members", "gets_credit": False},
}


def segment_action(
    age,
    visits_per_month,
    avg_visit_minutes,
    classes_per_month,
    months_member,
    monthly_extras_spend,
):
    row = pd.DataFrame(
        [[
            age,
            visits_per_month,
            avg_visit_minutes,
            classes_per_month,
            months_member,
            monthly_extras_spend,
        ]],
        columns=features,
    )
    row_scaled = scaler.transform(row)
    cluster = int(final_model.predict(row_scaled)[0])
    segment = segments[cluster]
    return cluster, segment["segment_name"], segment["gets_credit"]


new_cluster, new_segment, new_credit = segment_action(
    age=30,
    visits_per_month=4,
    avg_visit_minutes=42,
    classes_per_month=1,
    months_member=5,
    monthly_extras_spend=10,
)

print()
print(f"New member cluster: {new_cluster}")
print(f"Segment name: {new_segment}")
print(f"Credit decision: {'gets the $10 credit' if new_credit else 'does not get the $10 credit'}")


def monthly_net(extras_spend, gets_credit):
    if not gets_credit:
        return 0
    if extras_spend < 20:
        return 5
    return -10


members = pd.concat([training, test], ignore_index=True)
members_scaled = scaler.transform(members[features])
members["cluster"] = final_model.predict(members_scaled)
members["segment_name"] = members["cluster"].map(lambda c: segments[c]["segment_name"])
members["gets_credit"] = members["cluster"].map(lambda c: segments[c]["gets_credit"])
members["v1_net"] = [
    monthly_net(extras, credit)
    for extras, credit in zip(members["monthly_extras_spend"], members["gets_credit"])
]
members["baseline_net"] = [
    monthly_net(extras, True) for extras in members["monthly_extras_spend"]
]

baseline_credit_count = len(members)
baseline_net = int(round(members["baseline_net"].sum()))
v1_credit_count = int(members["gets_credit"].sum())
v1_net = int(round(members["v1_net"].sum()))
improvement = v1_net - baseline_net

print()
print("BASELINE")
print(f"members receiving credit: {baseline_credit_count}")
print(f"monthly net: ${baseline_net}")
print()
print("V1")
print(f"members receiving credit: {v1_credit_count}")
print(f"monthly net: ${v1_net}")
print()
print("COMPARISON")
print(f"monthly improvement of V1 over baseline: ${improvement}")

summary_rows = []
for cluster in range(final_k):
    group = members[members["cluster"] == cluster]
    summary_rows.append({
        "cluster": cluster,
        "segment_name": segments[cluster]["segment_name"],
        "members": len(group),
        "avg_extras_spend": round(group["monthly_extras_spend"].mean(), 1),
        "gets_credit": segments[cluster]["gets_credit"],
        "v1_monthly_net": int(round(group["v1_net"].sum())),
    })

summary = pd.DataFrame(summary_rows)
print()
print(summary.to_string(index=False))

plan_names = ["Baseline", "V1"]
plan_nets = [baseline_net, v1_net]
bars = plt.bar(plan_names, plan_nets, color=["#8d8d8d", "#2f6fad"], width=0.55)
plt.axhline(0, color="black", linewidth=0.8)
plt.ylabel("Monthly Net ($)")
plt.title("Monthly Value of the Extras Credit: Baseline vs V1")

label_pad = max(abs(net) for net in plan_nets) * 0.04
for bar, net in zip(bars, plan_nets):
    if net < 0:
        label = f"-${abs(net):,.0f}"
        y = net - label_pad
        va = "top"
    else:
        label = f"${net:,.0f}"
        y = net + label_pad
        va = "bottom"
    plt.text(bar.get_x() + bar.get_width() / 2, y, label, ha="center", va=va)

low = min(0, min(plan_nets)) - label_pad * 6
high = max(0, max(plan_nets)) + label_pad * 6
plt.ylim(low, high)
plt.tight_layout()
plt.savefig("monthly_net_comparison.png")
plt.close()
