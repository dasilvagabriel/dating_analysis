# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 23:10:19 2023

@author: gabri
"""

import pandas as pd
import numpy as np
from app_store_scraper import AppStore

class AppReviewsScraper:
    """
    A class to scrape app reviews from the Apple App Store.

    Parameters
    ----------
    country : str
        Abbreviation in English of the country you want to scrape the reviews.
    app_name : str
        The name of the app.
    app_id : int
        The app id on the app store. To find this id, just google app_name on app store.
        The end of the url says id followed by the id number.
    how_many : int
        The number of reviews you want to scrape from the app store.

    Attributes
    ----------
    country : str
        Abbreviation in English of the country you want to scrape the reviews.
    app_name : str
        The name of the app.
    app_id : int
        The app id on the app store.
    how_many : int
        The number of reviews you want to scrape from the app store.
    app_store : AppStore
        An instance of the AppStore class from the app_store_scraper package.

    Methods
    -------
    scrape_reviews():
        Scrapes reviews for the given app from the App Store.
        Returns a Pandas DataFrame containing the reviews and their associated information.
    
    export_to_csv(file_path):
        Exports the scraped reviews to a CSV file at the given file path.
    """
    def __init__(self, country, app_name, app_id, how_many):
        self.country = country
        self.app_name = app_name
        self.app_id = app_id
        self.how_many = how_many
        self.app_store = AppStore(country=self.country, app_name=self.app_name, app_id=self.app_id)
        
    def scrape_reviews(self):
        self.app_store.review(how_many=self.how_many)
        reviews_df = pd.DataFrame(np.array(self.app_store.reviews), columns=['review'])
        reviews_df = reviews_df.join(pd.DataFrame(reviews_df.pop('review').tolist()))
        return reviews_df
    
    def export_to_csv(self, file_path):
        reviews_df = self.scrape_reviews()
        reviews_df.to_csv(file_path, index=False)

# instantiate the class for Grindr app reviews
grindr_scraper = AppReviewsScraper(country='us', app_name='grindr', app_id='319881193', how_many=2000)

# scrape the reviews and export them to a CSV file
grindr_scraper.export_to_csv('grindr_reviews.csv')

# instantiate the class for Scruff app reviews
scruff_scraper = AppReviewsScraper(country='us', app_name='scruff', app_id='380015247', how_many=2000)

# scrape the reviews and export them to a CSV file
scruff_scraper.export_to_csv('scruff_reviews.csv')

# instantiate the class for Hornet app reviews
hornet_scraper = AppReviewsScraper(country='us', app_name='hornet', app_id='462678375', how_many=2000)

# scrape the reviews and export them to a CSV file
hornet_scraper.export_to_csv('hornet_reviews.csv')