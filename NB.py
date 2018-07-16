import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import math
import pandas as pd
import csv

### Load training data
training_data = []

with open('base_treino_secretaria.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=';')
    for row in readCSV:
        training_data.append({"class": row[0], "sentence": row[1]})

# Stemmer for portuguese
stemmer = SnowballStemmer("portuguese")
stopWords = list(set(stopwords.words('portuguese')))

#All classes
classes = list()
classes = list(set([classe['class'] for classe in training_data]))

#Calculate probabilities for each class
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

dicionario_1 = dict()
for sentenca in training_data:
    # tokenizar cada sentenca

    if x != sentenca['class']:
        x = sentenca['class']
        dicionario_2 = dict()

    for word in nltk.word_tokenize(sentenca['sentence']):

        if word not in (stopWords + ['.',',','?']):
            # stemming
            stemmed_word = stemmer.stem(word.lower())

            if stemmed_word not in dicionario_2:
                dicionario_2[stemmed_word] = 1
            else:
                dicionario_2[stemmed_word] += 1

        dicionario_1[x] = dicionario_2

#Suavization
suavizacao = dicionario_1

for intencao in suavizacao:
    for word in suavizacao[intencao]:
         suavizacao[intencao][word] += 1

#Add vocabulary
for sentenca in training_data:
    # tokenizing
    for word in nltk.word_tokenize(sentenca['sentence']):
        if word not in (stopWords + ['.',',','?']):
        # stemming
            stemmed_word = stemmer.stem(word.lower())

        for classe in classes:

            #add vocabulary

            if stemmed_word not in suavizacao[classe]:
                suavizacao[classe][stemmed_word] = 1

# Calcular score de uma sentenca nova

def naive_bayes(sentenca, show_details=False):
    '''
    Receive a sentence and calculate the most probable class using NB modeling.
    :setenca type: string
    :show_details type: bool (default = False)
    :r type: string
    '''
    import math
    probabilidades = dict()
    for intencao in classes:

        score = 0.0
        somatorio = 0

        # tokenize each word in our new sentence
        for word in nltk.word_tokenize(sentenca):

            # verificar se a palavra está em alguma classe
            stemmed = stemmer.stem(word.lower())
            if stemmed in suavizacao[intencao]:

                # p(c)*p(atr/c)

                prob_atr = suavizacao[intencao][stemmed]/sum(suavizacao[intencao].values())
                somatorio = somatorio + math.log10(prob_atr)

        probabilidades[intencao] = math.log10(prob_c[intencao]) + somatorio

        from collections import OrderedDict
        sorted_by_value = OrderedDict(sorted(probabilidades.items(),
                                  key=lambda kv: kv[1], reverse=True))

    if show_details is True:
        print(sorted_by_value)

    return list(sorted_by_value)[0]
