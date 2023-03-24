# %%
import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score, classification_report
import re
from nltk.tokenize import word_tokenize
from nltk.tokenize.casual import TweetTokenizer

import unicodedata
import emoji

def cleanTweet_TweetTokenizer(tweet):

    clean_tweet = re.sub(r'[.]', ' . ', clean_tweet) # Add spaces around periods

    clean_tweet = re.sub(r"([a-zA-Z])[´`]([a-zA-Z])", "\g<1>'\g<2>", clean_tweet) # Convert ´ and ` when surrounded by letters
    clean_tweet = re.sub(r'[´`^¨~|]', '', clean_tweet) # Remove special characters

    clean_tweet = ' '.join(TweetTokenizer(strip_handles = False, reduce_len = True, preserve_case = False).tokenize(clean_tweet)) # Tokenise tweet

    #clean_tweet = re.sub(r'@[a-z_]+', '<USERNAME>', clean_tweet)

    clean_tweet = re.sub(r'<3', emoji.emojize(':red_heart:'), clean_tweet) # Convert <3 into an emoji
    clean_tweet = re.sub(r'[<>]', '', clean_tweet) # Remove < and >

    clean_tweet = re.sub("http[^\s]+", '', tweet) # Remove links

    clean_tweet = re.sub(r'@[^\s]+', '<USERNAME>', clean_tweet) # Replace usernames with <USERNAME>

    print(f'Original Tweet: {tweet}\nCleaned Tweet: {clean_tweet}\n')

    return clean_tweet

# %%
df = pd.read_json('./EXIST2023_training.json', encoding='utf8', orient = 'index')

# %%
X_train = df['tweet'].to_list()
X_train_lang = ['english' if lang == 'en' else 'spanish' for lang in df['lang'].to_list()] # Change language labels to full name
Y_train = ['YES' if labels.count('YES') > 3 else 'NO' for labels in df['labels_task1'].to_list()] # Label as sexism if 3 or more annotators label the tweet as sexism
X_train_new = [cleanTweet_TweetTokenizer(tweet) for tweet in X_train]

## %%
## create TF-IDF vectorizer with n-grams
#tfidf = TfidfVectorizer(ngram_range=(2, 2), analyzer='char')
#
## fit and transform the training data
#X_train_tfidf = tfidf.fit_transform(X_train)
#
## transform the testing data using the same vectorizer
#X_test_tfidf = tfidf.transform(X_test)
#
## %%
## create SVM classifier and fit the training data
#svm = SVC(kernel='rbf')
#svm.fit(X_train_tfidf, Y_train)
#
## predict on the testing data and calculate accuracy
#y_pred = svm.predict(X_test_tfidf)
#accuracy = accuracy_score(Y_test, y_pred)
#print('Accuracy:', accuracy*100)
#
## print classification report
#print(classification_report(Y_test, y_pred))
