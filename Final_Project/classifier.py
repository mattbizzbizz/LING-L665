# %%
import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score, classification_report
import re
from nltk.tokenize import word_tokenize
#from nltk.tokenize.casual import TweetTokenizer
from TweetTokenizer_modified import TweetTokenizer

import unicodedata
import emoji

import pickle

def cleanTweet(tweet):

    clean_tweet = re.sub(r'https://[a-zA-Z0-9/.]+', '', tweet) # Remove links

    clean_tweet = re.sub(r'[.]', ' . ', clean_tweet) # Add spaces around periods

    clean_tweet = re.sub(r"([a-zA-Z])[´`]([a-zA-Z])", "\g<1>'\g<2>", clean_tweet) # Convert ´ and ` when surrounded by letters
    clean_tweet = re.sub(r'([0-9])°', '\g<1> degrees', clean_tweet) # Convert ° into the word 'degrees' when directly after a number
    clean_tweet = re.sub(r'[´`^¨~°|─­]', '', clean_tweet) # Remove special characters

    clean_tweet = ' '.join(TweetTokenizer(strip_handles = True, reduce_len = True, preserve_case = False).tokenize(clean_tweet)) # Tokenise tweet

    clean_tweet = re.sub(r'<3', emoji.emojize(':red_heart:'), clean_tweet) # Convert <3 into an emoji
    clean_tweet = re.sub(r'[<>]', '', clean_tweet) # Remove < and >

    #print(f'Original Tweet: {tweet}\nCleaned Tweet: {clean_tweet}\n')

    return clean_tweet

# %%
df = pd.read_json('./EXIST2023_training.json', encoding='utf8', orient = 'index')

# %%
X_train = df['tweet'].to_list()
X_train_lang = ['english' if lang == 'en' else 'spanish' for lang in df['lang'].to_list()] # Change language labels to full name
Y_train = ['YES' if labels.count('YES') > 3 else 'NO' for labels in df['labels_task1'].to_list()] # Label as sexism if 3 or more annotators label the tweet as sexism
X_train_clean = [cleanTweet(tweet) for tweet in X_train]

X_train_spanish = []
X_train_english = []
Y_train_spanish_majority = []
Y_train_english_majority = []
Y_train_spanish = []
Y_train_english = []

for i, lang in enumerate(X_train_lang):
    if lang == 'english':
        X_train_english.append(X_train_clean[i])
        Y_train_english_majority.append(Y_train[i])
        Y_train_english.append(df['labels_task1'].to_list()[i])
    elif lang == 'spanish':
        X_train_spanish.append(X_train_clean[i])
        Y_train_spanish_majority.append(Y_train[i])
        Y_train_spanish.append(df['labels_task1'].to_list()[i])
    else:
        print("Invalid language name in X_train_lang.")
        exit()

with open('tokenizer_english.pkl', 'wb') as fd:
    pickle.dump(X_train_english, fd)

with open('tokenizer_spanish.pkl', 'wb') as fd:
    pickle.dump(X_train_spanish, fd)

with open('tokenizer_english_labels_majority.pkl', 'wb') as fd:
    pickle.dump(Y_train_english_majority, fd)

with open('tokenizer_spanish_labels_majority.pkl', 'wb') as fd:
    pickle.dump(Y_train_spanish_majority, fd)

with open('tokenizer_english_labels.pkl', 'wb') as fd:
    pickle.dump(Y_train_english, fd)

with open('tokenizer_spanish_labels.pkl', 'wb') as fd:
    pickle.dump(Y_train_spanish, fd)


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
