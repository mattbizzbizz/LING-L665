import unicodedata
from emoji import EMOJI_DATA

def is_emoji(s):
    return s in EMOJI_DATA

file = open("EXIST2023_training.json", "r")
#file = open("test.json", "r")

for f in file:

    line = f.lstrip()

    if line.startswith('\"tweet\"'):

        line = line.lstrip('\"tweet\": ') # Remove label name
        line = line.rstrip('\n') # Remove newline

        prev_char_category = ''
        unicode_categories_set = set()

        print('[', end = '')

        for word in line.split(' '):

            if word.startswith('http'):
                break

            for char in word:

                char_category = unicodedata.category(char)

                if 'Po' == char_category:

                    #print('\n\nInside Po check\n')

                    if is_emoji(char):

                        if ('L' in prev_char_category) or ('E' == prev_char_category) or ('So' == prev_char_category):
                            print(' ' + char, end = '')
                        else: print(char, end = '')

                        prev_char_category = 'E'

                    else: prev_char_category = char_category

                elif 'P' in char_category:

                    #print('\n\nInside P check\n')

                    if 'P' in prev_char_category:
                        print(' ', end = '')

                    prev_char_category = char_category

                else:

                    #print('\n\n---Else---\n')
                    #print(f'---Character: {char}')
                    #print(f'---Character Category: {char_category}')

                    print(char, end = '')
                    prev_char_category = char_category

                unicode_categories_set.add(unicodedata.category(char))

            print(' ', end = '')

        print('\t', end = '')

    elif line.startswith('\"labels_task1\"'):

        label_count = line.count('\"YES\"')

        if label_count > 3:
            print('YES]')
        elif label_count < 3:
            print('NO]')
        else:
            print('UNCLEAR]')

print(unicode_categories_set)
