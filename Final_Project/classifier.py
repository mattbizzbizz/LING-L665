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

#%%
train_data = pd.read_json('./EXIST_2023_Dataset/training/EXIST2023_training.json', encoding='utf8', orient = 'index') # Open training data json file
train_golds_soft = pd.read_json('./EXIST_2023_Dataset/evaluation/golds/EXIST2023_training_task1_gold_soft.json', encoding='utf8', orient = 'index') # Open training soft labels json file
X_train_ID = train_data['id_EXIST'].to_list() # Create list of training file tweet IDs
X_train_tweet = train_data['tweet'].to_list() # Create list of tweets
X_train_lang = ['english' if lang == 'en' else 'spanish' for lang in train_data['lang'].to_list()] # Change language labels to full name
X_train_clean_tweet, X_train_username_counts, X_train_exclamation_counts, X_train_question_counts, X_train_possesive_username_counts = zip(*[cleanTweet(tweet) for tweet in X_train_tweet]) # Clean tweets and extract features
Y_train_soft = [[round(softs['YES'], 2), round(softs['NO'], 2)] for softs in train_golds_soft['soft_label'].to_list()] # Round soft labels to 2 decimal places

train_data_spanish = [] # List of tweet IDs and tweets for spanish
train_data_english = [] # List of tweet IDs and tweets for english
train_labels_spanish = [] # List of tweet IDs and soft labels for spanish
train_labels_english = [] # List of tweet IDs and soft labels for english

## Iterate through language labels
for i, lang in enumerate(X_train_lang):

    ## Check if language is english
    if lang == 'english':
        train_data_english.append([X_train_ID[i], X_train_clean_tweet[i]]) # Add tweet ID and tweet to list
        train_labels_english.append([X_train_ID[i], Y_train_soft[i]]) # Add tweet ID and soft label to list

    ## Check if language is spanish
    elif lang == 'spanish':
        train_data_spanish.append([X_train_ID[i], X_train_clean_tweet[i]]) # Add tweet ID and tweet to list
        train_labels_spanish.append([X_train_ID[i], Y_train_soft[i]]) # Add tweet ID and soft label to list

    ## If language is not english or spanish return error message and close program
    else:
        print("Invalid language name in X_train_lang.")
        exit()

## Save training data for english as pkl file
with open('./pkl/train_data_english.pkl', 'wb') as fd:
    pickle.dump(train_data_english, fd)

## Save training data for spanish as pkl file
with open('./pkl/train_data_spanish.pkl', 'wb') as fd:
    pickle.dump(train_data_spanish, fd)

## Save training labels for english as pkl file
with open('./pkl/train_labels_english.pkl', 'wb') as fd:
    pickle.dump(train_labels_english, fd)

## Save training labels for spanish as pkl file
with open('./pkl/train_labels_spanish.pkl', 'wb') as fd:
    pickle.dump(train_labels_spanish, fd)

#%%
dev_data = pd.read_json('./EXIST_2023_Dataset/dev/EXIST2023_dev.json', encoding='utf8', orient = 'index') # Open dev data json file
dev_golds_soft = pd.read_json('./EXIST_2023_Dataset/evaluation/golds/EXIST2023_dev_task1_gold_soft.json', encoding='utf8', orient = 'index') # Open dev soft labels json file
X_dev_ID = dev_data['id_EXIST'].to_list() # Create list of dev file tweet IDs
X_dev_tweet = dev_data['tweet'].to_list() # Create list of tweets
X_dev_lang = ['english' if lang == 'en' else 'spanish' for lang in dev_data['lang'].to_list()] # Change language labels to full name
X_dev_clean_tweet, X_dev_username_counts, X_dev_exclamation_counts, X_dev_question_counts, X_dev_possesive_username_counts = zip(*[cleanTweet(tweet) for tweet in X_dev_tweet]) # Clean tweets and extract features
Y_dev_soft = [[round(softs['YES'], 2), round(softs['NO'], 2)] for softs in dev_golds_soft['soft_label'].to_list()] # Round soft labels to 2 decimal places

dev_data_spanish = [] # List of tweet IDs and tweets for spanish
dev_data_english = [] # List of tweet IDs and tweets for english
dev_labels_spanish = [] # List of tweet IDs and soft labels for spanish
dev_labels_english = [] # List of tweet IDs and soft labels for english

## Iterate through language labels
for i, lang in enumerate(X_dev_lang):

    ## Check if language is english
    if lang == 'english':
        dev_data_english.append([X_dev_ID[i], X_dev_clean_tweet[i]]) # Add tweet ID and tweet to list
        dev_labels_english.append([X_dev_ID[i], Y_dev_soft[i]]) # Add tweet ID and soft label to list

    ## Check if language is spanish
    elif lang == 'spanish':
        dev_data_spanish.append([X_dev_ID[i], X_dev_clean_tweet[i]]) # Add tweet ID and tweet to list
        dev_labels_spanish.append([X_dev_ID[i], Y_dev_soft[i]]) # Add tweet ID and soft label to list

    ## If language is not english or spanish return error message and close program
    else:
        print("Invalid language name in X_dev_lang.")
        exit()

## Save dev data for english as pkl file
with open('./pkl/dev_data_english.pkl', 'wb') as fd:
    pickle.dump(dev_data_english, fd)

## Save dev data for spanish as pkl file
with open('./pkl/dev_data_spanish.pkl', 'wb') as fd:
    pickle.dump(dev_data_spanish, fd)

## Save dev labels for english as pkl file
with open('./pkl/dev_labels_english.pkl', 'wb') as fd:
    pickle.dump(dev_labels_english, fd)

## Save dev labels for spanish as pkl file
with open('./pkl/dev_labels_spanish.pkl', 'wb') as fd:
    pickle.dump(dev_labels_spanish, fd)

#%%
test_data = pd.read_json('./EXIST_2023_Dataset/test/EXIST2023_test_clean.json', encoding='utf8', orient = 'index') # Open test data json file
X_test_ID = test_data['id_EXIST'].to_list() # Create list of test file tweet IDs
X_test_tweet = test_data['tweet'].to_list() # Create list of tweets
X_test_lang = ['english' if lang == 'en' else 'spanish' for lang in test_data['lang'].to_list()] # Change language labels to full name
X_test_clean_tweet, X_test_username_counts, X_test_exclamation_counts, X_test_question_counts, X_test_possesive_username_counts = zip(*[cleanTweet(tweet) for tweet in X_test_tweet]) # Clean tweets and extract features

test_data_spanish = [] # List of tweet IDs and tweets for spanish
test_data_english = [] # List of tweet IDs and tweets for english

## Iterate through language labels
for i, lang in enumerate(X_test_lang):

    ## Check if language is english
    if lang == 'english':
        test_data_english.append([X_test_ID[i], X_test_clean_tweet[i]]) # Add tweet ID and tweet to list

    ## Check if language is spanish
    elif lang == 'spanish':
        test_data_spanish.append([X_test_ID[i], X_test_clean_tweet[i]]) # Add tweet ID and tweet to list

    ## If language is not english or spanish return error message and close program
    else:
        print("Invalid language name in X_test_lang.")
        exit()

## Save test data for english as pkl file
with open('./pkl/test_data_english.pkl', 'wb') as fd:
    pickle.dump(test_data_english, fd)

## Save test data for spanish as pkl file
with open('./pkl/test_data_spanish.pkl', 'wb') as fd:
    pickle.dump(test_data_spanish, fd)
