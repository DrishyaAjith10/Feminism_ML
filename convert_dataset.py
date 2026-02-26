import pandas as pd

# Load dev dataset
data = pd.read_csv("dev.csv")

# Keep only necessary columns
data = data[["text", "label_sexist"]]

# Map labels
data["label"] = data["label_sexist"].map({
    "not sexist": 1,
    "sexist": 0
})

# Drop original label column
data = data.drop(columns=["label_sexist"])

# Rename text column
data = data.rename(columns={"text": "sentence"})

# Remove missing values
data = data.dropna()

# Shuffle dataset
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Save final dataset
data.to_csv("dataset.csv", index=False)

print("✅ Dataset converted successfully")
print("\nLabel distribution:")
print(data["label"].value_counts())