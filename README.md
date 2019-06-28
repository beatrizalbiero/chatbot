# READ ME

The simplest chatbot ever.
This first version of my chatbot is capable of answering questions about this web [page](http://pos.fflch.usp.br/).
(PT-BR)

---
# Requirements

```
$ sudo apt-get install python3-pip
$ pip3 install -r requirements.txt
```

---
# The challenge

- Build a chatbot that is campable of identifying a few possible intentions
- Do this using a very short training data set.

---
# Models

- SVM (with sk-learn)
- Logistic Regression (with sk-learn)
- Naive Bayes from scratch! [here](https://github.com/beatrizalbiero/chatbot/blob/master/NB.py)

(It is also very simple to add different models)

---
# Data

Traning data:

base_treino_secretaria.csv

Testing data:

base_teste_secretaria.csv

Answers:

respostas_bot.csv

---
# Test it

```
$ python bot.py
```

# Models evaluation

```
$ python metrics.py
```
