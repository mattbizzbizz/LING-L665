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

import html

# Remove duplicated word
#     Return cleaned tweet and count of word
def remove_duplicate(tweet, target):

    # Split input string separated by space
    tweet = tweet.split(" ")

    target_bool = False # Bool for whether the target exists in the tweet
    count = 0 # Count of targets
    clean_tweet = '' # Tweet with additional targets removed

    # Iterate through words in tweet
    for word in tweet:

        # Check if word is target
        #     If it is not, add word to clean_tweet
        if word == target:

            # Increment count if 2+ target words were present
            #     otherwise, add target to clean_tweet and set target_bool to True
            if target_bool:
                count += 1
            else:
                count = 1
                clean_tweet += ' ' + word
                target_bool = True

        else:
            clean_tweet += ' ' + word

    return clean_tweet.strip(), count

# Replace upside-down punctation marks
#     Return cleaned tweet
def replace_punct(tweet, upside_down_punct, punct):

    # Split input string separated by space
    tweet = tweet.split(" ")

    clean_tweet = '' # Tweet with additional targets removed

    stack = [] # Stack of punctuation marks

    # Iterate through words in tweet
    for word in tweet:

        # Check if word is target
        #     If it is not, add word to clean_tweet
        if word == upside_down_punct: stack.append(upside_down_punct)

        elif word == punct:

            if stack:
                stack.pop()
                clean_tweet += ' ' + word
            else: clean_tweet += ' ' + word

        else: clean_tweet += ' ' + word

    for elem in stack:
        clean_tweet += ' ' + punct

    return clean_tweet.strip()

def normalize(tweet):
    clean_tweet = ''
    for char in tweet:
        val = ord(char)
        if val >= 119938 and val <= 120067:
            val -= 119841
        clean_tweet += chr(val)
    return clean_tweet

def cleanTweet(tweet):

    clean_tweet = html.unescape(tweet) # Convert html characters to unicode
    clean_tweet = unicodedata.normalize('NFKC', clean_tweet) # Normalize font
    clean_tweet = normalize(clean_tweet) # Fix weird fonts

    clean_tweet = re.sub(r'•͈ᴗ•͈', emoji.emojize(':smiling_face_with_tear:'), clean_tweet) # Convert •͈ᴗ•͈ into an emoji

    clean_tweet = re.sub(r'https://[a-zA-Z0-9/.:]+', '', clean_tweet) # Remove links

    clean_tweet = re.sub(r'(@[a-zA-Z]+)@([a-zA-Z])', '\g<1> @\g<2>', clean_tweet) # Add space between usernames

    clean_tweet = re.sub(r'([^ @][a-zA-z])@([a-zA-Z])', '\g<1>ATIDENTIFICATIONTAG\g<2>', clean_tweet) # Turn @ into an identification tag if it is not a username

    clean_tweet = re.sub(r"([a-zA-Z ])[´`‘’]([a-zA-Z])", "\g<1>'\g<2>", clean_tweet) # Convert ´ and ` when surrounded by letters
    clean_tweet = re.sub(r'([0-9])°', '\g<1> degrees', clean_tweet) # Convert ° into the word 'degrees' when directly after a number
    clean_tweet = re.sub(r'([0-9])%', '\g<1> percent', clean_tweet) # Convert % into the word 'percent' when directly after a number
    clean_tweet = re.sub(r'([a-zA-Z])[*]+([a-z])', '\g<1>astrickidentificationtag\g<2>', clean_tweet) # Convert censoring astricks into identification tags

    clean_tweet = re.sub(r'[.´`^¨~°|─­,;‘’"“”«»()\[\]{}®\$£€*#%↓ِ\u0301\u200D]', ' ', clean_tweet) # Replace special characters with a space

    clean_tweet = re.sub(r'\u00A9\uFE0F', 'c', clean_tweet) # DOUBLE-CHECK THIS Replacing copyrite symbol with a 'c'

    clean_tweet = ' '.join(TweetTokenizer(strip_handles = True, reduce_len = True, preserve_case = False).tokenize(clean_tweet)) # Tokenise tweet

    clean_tweet = re.sub(r' :($|\s)', '\g<1> ', clean_tweet) # Replace colons with a space when they aren't part of a time

    clean_tweet = re.sub(r' / ', ' ', clean_tweet) # Remove backslashes
    clean_tweet = re.sub(r'\w*\d\w*', ' ', clean_tweet) # Remove words with numbers

    clean_tweet = re.sub(r'<3', emoji.emojize(':red_heart:'), clean_tweet) # Convert <3 into an emoji
    clean_tweet = re.sub(r'\+', ' plus ', clean_tweet) # Convert + into the word plus
    clean_tweet = re.sub(r'\-', ' minus ', clean_tweet) # Convert - into the word minus

    clean_tweet = re.sub(r'atidentificationtag', '@', clean_tweet) # Convert @ symbols back
    clean_tweet = re.sub(r'astrickidentificationtag', '*', clean_tweet) # Convert * symbols back

    clean_tweet = re.sub(r'[<>]', '', clean_tweet) # Remove < and >
    clean_tweet = re.sub(r' -($|\s)', ' ', clean_tweet) # Remove hyphens when not connecting words or numbers

    clean_tweet = re.sub(r'usernameidentificationtag', '<USERNAME>', clean_tweet) # Convert usernames to <USERNAME>

    clean_tweet = re.sub(r"([a-z>]) '[\s]*s ", "\g<1>'s ", clean_tweet) # Reattach possesives

    clean_tweet, username_count = remove_duplicate(clean_tweet, '<USERNAME>') # Remove duplicate <USERNAME>
    clean_tweet, possesive_username_count = remove_duplicate(clean_tweet, "<USERNAME>'s") # Remove duplicate <USERNAME>'s

    clean_tweet = replace_punct(clean_tweet, '¡', '!') # Convert upside-down exclamation points to exclamation points
    clean_tweet = replace_punct(clean_tweet, '¿', '?') # Convert upside-down question marks to question marks
    clean_tweet = re.sub(r'&', 'and', clean_tweet) # Convert ampersand to the word 'and'
    clean_tweet = re.sub(r'à', 'á', clean_tweet) # Convert à to á
    clean_tweet = re.sub(r'ª', 'a', clean_tweet) # Convert ª to a
    clean_tweet = re.sub(r'[êė]', 'e', clean_tweet) # Convert ê to e
    clean_tweet = re.sub(r'ò', 'ó', clean_tweet) # Convert ò to ó
    clean_tweet = re.sub(r'ô', 'o', clean_tweet) # Convert ô to o

    clean_tweet, exclamation_count = remove_duplicate(clean_tweet, '!') # Remove duplicate exclamation points
    clean_tweet, question_count = remove_duplicate(clean_tweet, '?') # Remove duplicate exclamation points
    clean_tweet = re.sub(r'[\u0600-\u06FF]', '', clean_tweet) # Remove Arabic characters
    clean_tweet = re.sub(r'[\u10A0-\u10FF]+', '', clean_tweet) # Remove Gregorian characters
    clean_tweet = re.sub(r'[\u4E00-\u9FFF]+', '', clean_tweet) # Remove CJK characters
    clean_tweet = re.sub(r'[\uAC00-\uD7AF]+', '', clean_tweet) # Remove Hangul characters
    clean_tweet = re.sub(r'[\u3040-\u309F]+', '', clean_tweet) # Remove hiragana
    clean_tweet = re.sub(r" '()", " ", clean_tweet) # Remove separated apostrophes
    clean_tweet = re.sub(r' +', ' ', clean_tweet) # Remove double-spaces

    return clean_tweet, username_count, exclamation_count, question_count, possesive_username_count

df = pd.read_json('./EXIST2023_training.json', encoding='utf8', orient = 'index')
X_train = df['tweet'].to_list()
X_train_lang = ['english' if lang == 'en' else 'spanish' for lang in df['lang'].to_list()] # Change language labels to full name
Y_train = ['YES' if labels.count('YES') > 3 else 'NO' for labels in df['labels_task1'].to_list()] # Label as sexism if 3 or more annotators label the tweet as sexism
X_train_clean, X_train_username_counts, X_train_exclamation_counts, X_train_question_counts, X_train_possesive_username_counts = zip(*[cleanTweet(tweet) for tweet in X_train])

## Model will replace this section:
#X_train_spanish = []
#X_train_english = []
#Y_train_spanish_majority = []
#Y_train_english_majority = []
#Y_train_spanish = []
#Y_train_english = []
#
#for i, lang in enumerate(X_train_lang):
#    if lang == 'english':
#        X_train_english.append(X_train_clean[i])
#        Y_train_english_majority.append(Y_train[i])
#        Y_train_english.append(df['labels_task1'].to_list()[i])
#    elif lang == 'spanish':
#        X_train_spanish.append(X_train_clean[i])
#        Y_train_spanish_majority.append(Y_train[i])
#        Y_train_spanish.append(df['labels_task1'].to_list()[i])
#    else:
#        print("Invalid language name in X_train_lang.")
#        exit()
#
#with open('tokenizer_english.pkl', 'wb') as fd:
#    pickle.dump(X_train_english, fd)
#
#with open('tokenizer_spanish.pkl', 'wb') as fd:
#    pickle.dump(X_train_spanish, fd)
#
#with open('tokenizer_english_labels_majority.pkl', 'wb') as fd:
#    pickle.dump(Y_train_english_majority, fd)
#
#with open('tokenizer_spanish_labels_majority.pkl', 'wb') as fd:
#    pickle.dump(Y_train_spanish_majority, fd)
#
#with open('tokenizer_english_labels.pkl', 'wb') as fd:
#    pickle.dump(Y_train_english, fd)
#
#with open('tokenizer_spanish_labels.pkl', 'wb') as fd:
#    pickle.dump(Y_train_spanish, fd)


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
