from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import nltk
import csv

# Load training data
training_data = []

with open('base_treino_secretaria.csv', encoding='Latin-1') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=';')
    for row in readCSV:
        training_data.append({"class": row[0], "sentence": row[1]})

# Stemmer for portuguese
stemmer = SnowballStemmer("portuguese")
stopWords = list(set(stopwords.words('portuguese')))

# set of classes
classes = list()
classes = list(set([classe['class'] for classe in training_data]))

# Calculate probabilities for each class
prob_c = {}
for classe in classes:
        prob_c[classe] = 0

for data in training_data:
    for classe in classes:
        if data['class'] == classe:
            prob_c[classe] += 1
for classe in classes:
    prob_c[classe] = prob_c[classe]/len(training_data)

# Dictionary with the frequency of each word in each class
x = ''

dictio_1 = dict()
for sentence in training_data:
    # tokenizar cada sentence

    if x != sentence['class']:
        x = sentence['class']
        dictio_2 = dict()

    for word in nltk.word_tokenize(sentence['sentence']):

        if word not in (stopWords + ['.', ',', '?']):
            # stemming
            stemmed_word = stemmer.stem(word.lower())

            if stemmed_word not in dictio_2:
                dictio_2[stemmed_word] = 1
            else:
                dictio_2[stemmed_word] += 1

        dictio_1[x] = dictio_2

# Smoothing
smoothing = dictio_1

for intencao in smoothing:
    for word in smoothing[intencao]:
        smoothing[intencao][word] += 1

# Add vocabulary
for sentence in training_data:
    # tokenizing
    for word in nltk.word_tokenize(sentence['sentence']):
        if word not in (stopWords + ['.', ',', '?']):
            # stemming
            stemmed_word = stemmer.stem(word.lower())

        for classe in classes:
            # add vocabulary
            if stemmed_word not in smoothing[classe]:
                smoothing[classe][stemmed_word] = 1

# Calculate the score for a new sentence


def naive_bayes(sentence, show_details=False):
    """
    Receive a sentence and calculate the most probable class using NB modeling.

    :setenca type: string
    :show_details type: bool (default = False)
    :r type: string
    """
    import math
    probabilities = dict()
    for intencao in classes:

        somatorio = 0

        # tokenize each word in our new sentence
        for word in nltk.word_tokenize(sentence):

            stemmed = stemmer.stem(word.lower())
            if stemmed in smoothing[intencao]:

                # p(c)*p(atr/c)

                prob_atr = smoothing[intencao][stemmed] / \
                                            sum(smoothing[intencao].values())
                somatorio = somatorio + math.log10(prob_atr)

        probabilities[intencao] = math.log10(prob_c[intencao]) + somatorio

        from collections import OrderedDict
        sorted_by_value = OrderedDict(sorted(probabilities.items(),
                                             key=lambda kv: kv[1],
                                             reverse=True))

    if show_details is True:
        print(sorted_by_value)

    return list(sorted_by_value)[0]


def classification(interaction):
    """
    Classify an interaction.

    :interaction type: str
    :rtype: list
    """
    return naive_bayes(interaction)
