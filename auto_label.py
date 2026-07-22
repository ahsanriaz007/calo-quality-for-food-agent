import pandas as pd

# Load your filtered reviews
df = pd.read_csv("labeled_reviews.csv")


# Define a rule-based auto-categorizer function
def assign_category(text):
  text = str(text).lower()

  if any(w in text for w in ["hair", "bug", "mold", "sick", "poison", "smell"]):
    return "food_safety"
  elif any(w in text for w in ["portion", "small", "little", "macro", "fill"]):
    return "portioning"
  elif any(w in text for w in ["wrong", "instead", "incorrect"]):
    return "wrong_item"
  elif any(w in text for w in ["missing", "forgot", "left out", "incomplete"]):
    return "missing_item"
  elif any(w in text for w in ["leak", "spill", "box", "package", "container"]):
    return "packaging"
  elif any(w in text for w in ["allergy", "allergic", "nut", "gluten"]):
    return "allergen"
  elif any(w in text for w in ["late", "driver", "cold", "hour", "delivery"]):
    return "delivery"
  else:
    return "other"


# Apply it automatically across the dataset
df["category"] = df["text"].apply(assign_category)

# Save the auto-labeled file back
df.to_csv("labeled_reviews.csv", index=False, encoding="utf-8-sig")
print("Successfully auto-labeled all reviews without manual typing!")