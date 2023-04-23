import re
import emoji
import pickle
import html
import pandas as pd
import unicodedata
from collections import Counter
import wordninja
from TweetTokenizer_modified import TweetTokenizer
import json

class Tokenize():
    
    def __init__(self):
        super().__init__()

        
    def run(self):
        
        ## ------------------------------------- Helper functions ------------------------------------------------------------##
        def read_pickle(file_name, path):
            with open(path+file_name, 'rb') as handle: 
                data = pickle.load(handle)
            return data
        
        def write_pickle(file_name, file_object, path):
            with open(path+file_name, 'wb') as file: 
                pickle.dump(file_object, file)
                
        def read_json(path, file_name):
            data = pd.read_json(path+file_name, encoding='utf8', orient = 'index')
            return data
        
        def write_txt(file_name, counter, path):
            f = open(path+file_name, 'w')
            for key, value in counter.most_common():
                f.write(str(key) + ' ' + str(value) + '\n')
            f.close()
        
        def write_pickle(path, file_name, data):
            with open(path+file_name, 'wb') as fd:
                pickle.dump(data, fd)

        def remove_duplicate(tweet, target):
            """
            Remove duplicated word
            Return cleaned tweet and count of word
            """   
            tweet = tweet.split(" ") # Split input string separated by space

            target_bool = False # Bool for whether the target exists in the tweet
            count = 0 # Count of targets
            clean_tweet = '' # Tweet with additional targets removed

            for word in tweet:   # Iterate through words in tweet    
                if word == target: # Check if word is target # If it is not, add word to clean_tweet
                    if target_bool: # Increment count if 2+ target words were present otherwise, add target to clean_tweet and set target_bool to True
                        count += 1
                    else:
                        count = 1
                        clean_tweet += ' ' + word
                        target_bool = True
                else:
                    clean_tweet += ' ' + word
            return clean_tweet.strip(), count
        
        def replace_punct(tweet, upside_down_punct, punct):
            """
            Replace upside-down punctation marks
            Return cleaned tweet
            """
            tweet = tweet.split(" ") # Split input string separated by space
            clean_tweet = '' # Tweet with additional targets removed
            stack = [] # Stack of punctuation marks
            
            for word in tweet:  # Iterate through words in tweet
                if word == upside_down_punct: stack.append(upside_down_punct) # Check if word is target. If it is not, add word to clean_tweet
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

#        def normalize_hashtags(tweet):
#
#            clean_tweet = ''
#
#            for word in tweet.split(' '):
#                if word == '':
#                    continue
#                if word[0] == '#':
#                    clean_tweet += ' ' + ' '.join(wordninja.split(word[1:]))
#                else:
#                    clean_tweet += ' ' + word
#
#            return clean_tweet
        
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

            clean_tweet = re.sub(r'[.´`^¨~°|─­,;‘’"“”«»()\[\]{}®\$£€*%↓ِ\u0301\u200D]', ' ', clean_tweet) # Replace special characters with a space
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

            hashtag_regex = re.compile('#[\w]+')
            hashtag_lst = hashtag_regex.findall(clean_tweet)

            #clean_tweet = normalize_hashtags(clean_tweet) # Parse hashtags
            clean_tweet = re.sub(r'#', '', clean_tweet) # Remove # symbols

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

            print(f'Original Tweet: {tweet}\nClean Tweet: {clean_tweet}\n')

            return clean_tweet, username_count, exclamation_count, question_count, possesive_username_count, hashtag_lst
        
        def prepare_data(data_path, gold_path, data_file, gold_file, mode):
            data = read_json(data_path, data_file) 
            if mode != 'test':
                golds_soft = read_json(gold_path, gold_file)

            X_ID = data['id_EXIST'].to_list() # Create list of training file tweet IDs
            X_tweet = data['tweet'].to_list() # Create list of tweets
            X_lang = ['english' if lang == 'en' else 'spanish' for lang in data['lang'].to_list()] # Change language labels to full name
            X_clean_tweet, X_username_counts, X_exclamation_counts, X_question_counts, X_possesive_username_counts, X_hashtag_lst = zip(*[cleanTweet(tweet) for tweet in X_tweet]) # Clean tweets and extract features
            if mode != 'test':
                Y_soft = [[round(softs['YES'], 2), round(softs['NO'], 2)] for softs in golds_soft['soft_label'].to_list()] # Round soft labels to 2 decimal places
                Y_hard = ['YES' if labels.count('YES') >= 3 else 'NO' for labels in data['labels_task1'].to_list()] # Create list of hard labels

            data_spanish = [] # List of tweet IDs and tweets for spanish
            data_english = [] # List of tweet IDs and tweets for english
            soft_labels_spanish = [] # List of tweet IDs and soft labels for spanish
            soft_labels_english = [] # List of tweet IDs and soft labels for english
            hard_labels_spanish = [] # List of tweet IDs and hard labels for spanish
            hard_labels_english = [] # List of tweet IDs and hard labels for english

            for i, lang in enumerate(X_lang): ## Iterate through language labels
                if lang == 'english': ## Check if language is english
                    data_english.append([X_ID[i], X_clean_tweet[i]]) # Add tweet ID and tweet to list
                    if mode != 'test':
                        soft_labels_english.append([X_ID[i], Y_soft[i]]) # Add tweet ID and soft label to list
                        hard_labels_english.append([X_ID[i], Y_hard[i]]) # Add tweet ID and hard label to list
                elif lang == 'spanish': ## Check if language is spanish
                    data_spanish.append([X_ID[i], X_clean_tweet[i]]) # Add tweet ID and tweet to list
                    if mode != 'test':
                        soft_labels_spanish.append([X_ID[i], Y_soft[i]]) # Add tweet ID and soft label to list
                        hard_labels_spanish.append([X_ID[i], Y_hard[i]]) # Add tweet ID and hard label to list
                else: ## If language is not english or spanish return error message and close program
                    print("Invalid language name in X_train_lang.")
                    exit()

            if mode != 'test':
                for i, hard_label in enumerate(Y_hard): ## Iterate through hard labels
                    if hard_label == 'YES': ## Check if the tweet is hard labeled as sexist
                        global_counter.update(X_hashtag_lst[i]) # Add hashtag list to counter

            ## writing files
            write_pickle('data/eng/raw/', f'{mode}_data.pkl', data_english)
            write_pickle('data/spa/raw/', f'{mode}_data.pkl', data_spanish)
            if mode != 'test': 
                write_pickle('data/eng/raw/', f'{mode}_soft_labels.pkl', soft_labels_english)
                write_pickle('data/spa/raw/', f'{mode}_soft_labels.pkl', soft_labels_spanish)
                write_pickle('data/eng/raw/', f'{mode}_hard_labels.pkl', hard_labels_english)
                write_pickle('data/spa/raw/', f'{mode}_hard_labels.pkl', hard_labels_spanish)
                
            return X_hashtag_lst
    


        ## ------------------------------------- Main Execution ------------------------------------------------------------##
        global_counter = Counter()
        
        ### --------------- Data ------------
        X_train_hashtag_lst = prepare_data('./EXIST_2023_Dataset/training/', './EXIST_2023_Dataset/evaluation/golds/', 'EXIST2023_training.json', 'EXIST2023_training_task1_gold_soft.json', 'train')
        X_dev_hashtag_lst = prepare_data('./EXIST_2023_Dataset/dev/', './EXIST_2023_Dataset/evaluation/golds/', 'EXIST2023_dev.json', 'EXIST2023_dev_task1_gold_soft.json', 'val')
        X_test_hashtag_lst = prepare_data('./EXIST_2023_Dataset/test/', '', 'EXIST2023_test_clean.json', '', 'test')
        
        
        ## writing txt files
        write_txt('hashtag_counter_sexist.txt', global_counter, 'data/')

        hashtag_lst = []
        for lst in X_train_hashtag_lst+X_dev_hashtag_lst+X_test_hashtag_lst:
            for elem in lst:
                hashtag_lst.append(elem)
        counter = Counter(hashtag_lst)
        write_txt('./hashtag_counter_all.txt', counter, 'data/')

tokenize = Tokenize()
tokenize.run()
