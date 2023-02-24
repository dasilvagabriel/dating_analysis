# Sentiment Analyzer
This code implements a sentiment analyzer for text reviews using a pre-trained BERT-based model. The code includes two classes: `SentimentAnalyzer` and `ReviewSentimentAnalyzer`.

## Libraries and Packages
The following libraries and packages are required for this code to run properly:

* pytorch
* torchvision
* torchaudio
* transformers
* requests
* beautifulsoup4
* pandas
* numpy

These can be installed using pip.

## Classes
### SentimentAnalyzer
This class takes a pre-trained BERT-based model and uses it to calculate the sentiment score of a given review text. The `SentimentAnalyzer` class has the following attributes:

* tokenizer: Instance of the AutoTokenizer class from transformers package for tokenizing input text.
* model: Instance of the AutoModelForSequenceClassification class from transformers package for sentiment analysis.

The `SentimentAnalyzer` class has the following methods:

* sentiment_score(review): Method for calculating sentiment score of a given review text using the pre-trained model.

### ReviewSentimentAnalyzer
This class takes a pre-trained BERT-based model and uses it to calculate the sentiment score of a collection of reviews given in a CSV file. The `ReviewSentimentAnalyzer` class has the following attributes:

* analyzer: Instance of the `SentimentAnalyzer` class for sentiment analysis of individual reviews.

The `ReviewSentimentAnalyzer` class has the following methods:

* analyze_reviews(review_csv_file_path): Method for analyzing sentiment of a collection of reviews given in a CSV file.

## Usage
1. First, install the required libraries and packages.
2. Next, load reviews from a CSV file into a pandas dataframe.
3. Apply the sentiment_score method to the 'review' column to add a new column called 'sentiment' to the dataframe.
4. Save the dataframe with the added sentiment column to a new CSV file.

Example:

```
#Load reviews from a CSV file into a pandas dataframe
hornet_reviews = pd.read_csv('hornet_reviews.csv')
#Add a new column called 'sentiment' to the dataframe by applying the sentiment_score method to the 'review' column
#The sentiment_score method uses a BERT model to predict the sentiment of the text, but only uses the first 512 characters
hornet_reviews['sentiment'] = df['review'].apply(lambda x: sentiment_score(x[:512]))
#Save the dataframe with the added sentiment column to a new CSV file
#The index column is excluded by setting index=False
hornet_reviews.to_csv('sentiment_hornet.csv', index=False)

```

## Debugging

* If you get a wordcloud related error, first, make sure that the wordcloud package is installed correctly. You can do this by running the following command in your terminal or command prompt:
`conda list wordcloud `
If the package is not installed, you can install it by running:
` conda install -c conda-forge wordcloud `
* If you face a "RemoveError" related to the requests package, it might be due to an outdated version of conda. To update conda, run the following command:
`conda update --force conda`
* If you encounter the error "RemoveError: 'requests' is a dependency of conda and cannot be removed from conda's operating environment," the suggested solution is updating conda by running:
` conda update --force conda `
* If you are running the code on a different Python IDE, you should note that the "!conda install" and "!pip install" commands should be run on the terminal, not in the Python script itself.