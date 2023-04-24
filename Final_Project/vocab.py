from collections import Counter

def counter_to_relative(counter):
    total_count = sum(counter.values())
    relative = Counter()
    for key in counter:
        relative[key] = counter[key] / total_count
    return relative

unique = Counter()
spa_unique = Counter()
eng_unique = Counter()

### SPANISH
for count in range(99):
    spa_file = open('spanish_billion_words/spanish_billion_words_' + '{:02}'.format(count), 'r')

    for line in spa_file:
        text = line.lower()
        words = text.split()
        words = [word.strip('.,!;()[]') for word in words]

        spa_unique.update(words)

### ENGLISH
with open('./unigram_freq.csv') as f:
    eng_file = f.readlines()[1:]


for line in eng_file:
    lst = line.split(',')
    key = lst[0]
    value = int(lst[1])
    eng_unique.update({key:value})

### COMBINE
unique = counter_to_relative(eng_unique) + counter_to_relative(spa_unique)

for pair in unique.most_common():
    print(pair[0])
