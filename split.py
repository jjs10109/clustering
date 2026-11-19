import pandas as pd
from sklearn.model_selection import train_test_split

members = pd.read_csv("gym_members.csv")

training_set, test_set = train_test_split(
    members,
    test_size=0.3,
    random_state=42,
)

training_set.to_csv("training_set.csv", index=False)
test_set.to_csv("test_set.csv", index=False)

print(f"total rows: {len(members)}")
print(f"training rows: {len(training_set)}")
print(f"test rows: {len(test_set)}")
