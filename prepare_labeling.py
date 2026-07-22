import pandas as pd

# 1. Load the raw scraped reviews
df = pd.read_csv("raw_reviews.csv")
print(f"Total raw reviews: {len(df)}")

# 2. Filter out missing text and very short reviews (less than 15 characters)
df = df.dropna(subset=["text"])
df["text_len"] = df["text"].astype(str).str.strip().str.len()
filtered_df = df[df["text_len"] >= 15].copy()

print(f"Reviews with substantial feedback (>= 15 chars): {len(filtered_df)}")

# 3. Add an empty 'category' column for your manual labeling
filtered_df["category"] = ""

# 4. Select the top 150 reviews to label
label_subset = filtered_df[["review_id", "rating", "date", "text", "category"]].head(150)

# 5. Save to a new file ready for manual labeling
label_subset.to_csv("labeled_reviews.csv", index=False, encoding="utf-8-sig")

print("Successfully created 'labeled_reviews.csv' with 150 rows ready for labeling!")