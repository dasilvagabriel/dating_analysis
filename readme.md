# Project README

This is an overview of each of the projects within this directory. Wthin each subdirectory you can find the appropriate readme.md, all the necessary files as well as the requirements.txt file. 

App Reviews Scraper:
The App Reviews Scraper code uses the app_store_scraper package to scrape reviews from the Apple App Store for a given app and export them to a CSV file. To use this code, install the app-store-scraper, pandas, and numpy libraries, and create an instance of the AppReviewsScraper class with the parameters for the country, app name, app id, and number of reviews to scrape. Then, call the export_to_csv method on the instance, passing a file path where the CSV file should be saved.

Sentiment Analyzer:
The Sentiment Analyzer code implements a sentiment analyzer for text reviews using a pre-trained BERT-based model. It includes two classes: SentimentAnalyzer and ReviewSentimentAnalyzer. To use this code, install the pytorch, torchvision, torchaudio, transformers, requests, beautifulsoup4, pandas, and numpy libraries. Load reviews from a CSV file into a pandas dataframe, apply the sentiment_score method to the 'review' column to add a new column called 'sentiment' to the dataframe, and save the dataframe with the added sentiment column to a new CSV file. If you encounter any errors, consult the debugging section of the README.