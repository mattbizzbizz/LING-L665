# %%
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import re
from nltk.tokenize import word_tokenize

# %%
def cleanTweet(tweet):
    clean_tweet = tweet.lower()
    clean_tweet = re.sub("@[A-Za-z0-9_]+", '', clean_tweet)
    clean_tweet = re.sub("\"", "", clean_tweet)
    clean_tweet = re.sub("http[://\/-zA-Z0-9_=+.$@#$%^&*()]+", '', clean_tweet) #Get rid of links
    
    clean_tweet = ' '.join(word_tokenize(clean_tweet))
    clean_tweet = re.sub("\'", "APOS", clean_tweet)
    clean_tweet = re.sub(" %", "PERC", clean_tweet)
    
    clean_tweet = re.sub("[^\w\s]", '', clean_tweet)
    clean_tweet = re.sub('APOS', "\'", clean_tweet)
    clean_tweet = re.sub("PERC", "%", clean_tweet)

    clean_tweet = re.sub(" +", " ", clean_tweet)
    clean_tweet = re.sub(" n't", "n't",  clean_tweet)
    clean_tweet = re.sub(" 's", "'s",  clean_tweet)

    
    return clean_tweet

# %%
train_df = pd.read_csv('./train.txt', encoding='utf8', sep='\t')
train_df.columns = ['tweet','label']


test_df = pd.read_csv('./test.txt', encoding='utf8', sep='\t')
test_df.columns = ['tweet','label']

validate_df = pd.read_csv('./validate.txt', encoding='utf8', sep='\t')
validate_df.columns = ['tweet','label']

# %%
X_train = train_df['tweet'].to_list()
Y_train = train_df['label'].to_list()
X_train_new = [cleanTweet(tweet) for tweet in X_train]

# %%
X_test = test_df['tweet'].to_list()
Y_test = test_df['label'].to_list()
X_test = [cleanTweet(tweet) for tweet in X_test]

# %%
# create TF-IDF vectorizer with n-grams
tfidf = TfidfVectorizer(ngram_range=(1, 2), analyzer='word')

# fit and transform the training data
X_train_tfidf = tfidf.fit_transform(X_train)

# transform the testing data using the same vectorizer
X_test_tfidf = tfidf.transform(X_test)

# %%
# create SVM classifier and fit the training data
svm = SVC(kernel='rbf')
svm.fit(X_train_tfidf, Y_train)

# predict on the testing data and calculate accuracy
y_pred = svm.predict(X_test_tfidf)
accuracy = accuracy_score(Y_test, y_pred)
print('Accuracy:', accuracy*100)

# print classification report
print(classification_report(Y_test, y_pred))
