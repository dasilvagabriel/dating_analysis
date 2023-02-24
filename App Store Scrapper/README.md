# App Reviews Scraper
This code uses the `app_store_scraper`package to scrape reviews from the Apple App Store for a given app, and export them to a CSV file. The package can be installed using `pip install app-store-scraper`.

## Usage
To use this code, create an instance of the `AppReviewsScraper` class with the following parameters:

* `country`: A string with the abbreviation in English of the country you want to scrape the reviews.
* `app_name`: A string with the name of the app.
* `app_id`: An integer with the app id on the app store. To find this `id`, just google app_name on app store. The end of the url says id followed by the id number.
* `how_many`: An integer with the number of reviews you want to scrape from the app store.

Then, call the `export_to_csv` method on the instance, passing a file path where the CSV file should be saved.

Example:

```
# instantiate the class for Scruff app reviews
scruff_scraper = AppReviewsScraper(country='us', app_name='scruff', app_id='380015247', how_many=2000)

# scrape the reviews and export them to a CSV file
scruff_scraper.export_to_csv('scruff_reviews.csv')
```

The above code will scrape 2000 reviews for the Scruff app in the US App Store, and export them to a file called `scruff_reviews.csv`.

## Class Details
### AppReviewsScraper Class
* `_init__(self, country, app_name, app_id, how_many)`: The class constructor. Initializes the AppStore instance from app_store_scraper with the specified parameters.
* `scrape_reviews(self)`: Scrapes reviews for the given app from the App Store. Returns a Pandas DataFrame containing the reviews and their associated information.
* `export_to_csv(self, file_path)`: Exports the scraped reviews to a CSV file at the given file path. Calls the scrape_reviews method to get the reviews.

## Requirements
* Python 3.6 or later
* app-store-scraper
* pandas
* numpy