from google_play_scraper import Sort, reviews_all
import pandas as pd

print("Fetching reviews for Calo app from Google Play...")

# Correct Calo Meal Subscription Package ID
app_id = "com.calo.webapp"

# Scrape all reviews
scraped_reviews = reviews_all(
    app_id,
    sleep_milliseconds=0,  # speed up download
    lang="en",  # fetch English reviews
    country="sa",  # Calo's primary market (Saudi Arabia)
    sort=Sort.NEWEST,
)

print(f"Downloaded {len(scraped_reviews)} reviews!")

# Format into a clean DataFrame
data = []
for r in scraped_reviews:
    data.append(
        {
            "review_id": r.get("reviewId"),
            "user_name": r.get("userName"),
            "rating": r.get("score"),
            "date": r.get("at"),
            "text": r.get("content"),
        }
    )

df = pd.DataFrame(data)

# Save to CSV in your project folder
csv_filename = "raw_reviews.csv"
df.to_csv(csv_filename, index=False, encoding="utf-8-sig")

print(f"Saved successfully to '{csv_filename}'!")