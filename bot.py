from LRmodel import classification
#from SVMmodel import classification
#from NB import classification
import csv
import random
from sklearn.feature_extraction.text import CountVectorizer

def load_answers(path):
    """
    Load answer base.

    :path type: str
    :rtype: dict
    """
    answers = dict()
    with open(path) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=';')
        for row in readCSV:
            answers[row[0]] = row[1:]
    return answers


def reply(intention, answers):
    """
    Replies to the user.

    :intention type: str
    :answers type: dict
    :rtype: print
    """
    import random

    n = len(answers[intention])

    if intention == 'saudacao' or intention == 'despedida':
        return print(answers[intention][random.randrange(n)])
    else:
        return print(answers[intention][random.randrange(n)] + " Posso ajudar com mais alguma coisa?")

def main():
    path = 'respostas_bot.csv'
    answers = load_answers(path)

    intention = ''

    print("Ola. Seja bem vindo ao canal de atendimento da secretaria. Como podemos te ajudar?")

    while intention != 'despedida':
        question = input()
        if question in ['não','n','nao']:
            print("Ok, ate breve!")
            break
        else:
            intention = classification(question)
            reply(intention, answers)

main()
