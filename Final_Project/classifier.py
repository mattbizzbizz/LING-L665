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
from emoji import EMOJI_DATA

def is_emoji(s):
    return s in EMOJI_DATA

# %%
def cleanTweet(tweet, lang):

    clean_tweet = tweet.lower() # Make tweet all lowercase

    clean_tweet = re.sub("@[A-Za-z0-9_]+", '', clean_tweet) # Remove usernames
    clean_tweet = re.sub("\"", "", clean_tweet) # Remove double-quotes
    clean_tweet = re.sub("http[://\/-zA-Z0-9_=+.$@#$%^&*()]+", '', clean_tweet) # Remove links
    
    clean_tweet = ' '.join(word_tokenize(clean_tweet, language = lang)) # Tokenise tweet

    clean_tweet = re.sub("\'", "APOS", clean_tweet) # Change apostrophes to a tag
    clean_tweet = re.sub(" %", "PERC", clean_tweet) # Change percent signs to a tag
    
    clean_tweet = re.sub("[^\w\s]", '', clean_tweet) # Remove punctuation

    clean_tweet = re.sub('APOS', "\'", clean_tweet) # Change apostrophe tag back to apostrophes
    clean_tweet = re.sub("PERC", "%", clean_tweet) # Change percent sign tags back to percent signs

    clean_tweet = re.sub(" +", " ", clean_tweet) # Remove double-spaces
    clean_tweet = re.sub(" n't", "n't",  clean_tweet) # Convert n't to not
    clean_tweet = re.sub(" 's", "'s",  clean_tweet) # Merge 's back into the previous word

    
    return clean_tweet

def cleanTweet_TweetTokenizer(tweet):

    clean_tweet = re.sub("http[://\/-zA-Z0-9_=+.$@#$%^&*()]+", '', tweet) # Remove links

    clean_tweet = re.sub(r'[.]', ' . ', clean_tweet)

    clean_tweet = re.sub(r"([a-z])`s", "\g<1>'s", clean_tweet)
    clean_tweet = re.sub(r'`', '', clean_tweet)

    clean_tweet = ' '.join(TweetTokenizer(strip_handles = True, reduce_len = True, preserve_case = False).tokenize(clean_tweet)) # Tokenise tweet

    print(f'Original Tweet: {tweet}\nCleaned Tweet: {clean_tweet}\n')

    return clean_tweet

# %%
df = pd.read_json('./EXIST2023_training.json', encoding='utf8', orient = 'index')

# %%
X_train = df['tweet'].to_list()
X_train_lang = ['english' if lang == 'en' else 'spanish' for lang in df['lang'].to_list()] # Change language labels to full name
Y_train = ['YES' if labels.count('YES') > 3 else 'NO' for labels in df['labels_task1'].to_list()] # Label as sexism if 3 or more annotators label the tweet as sexism
X_train_new = [cleanTweet(tweet, lang) for tweet, lang in zip(X_train, X_train_lang)]
X_train_new2 = [cleanTweet_TweetTokenizer(tweet) for tweet in X_train]

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
