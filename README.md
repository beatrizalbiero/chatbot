# READ ME

After weeks developing chatbots using stupid paid software, I've decided I've had enough of this and built a simple, yet my OWN chatbot.

I really mean it, it is really simple. But it was a cool exercise!

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
This first version of my chatbot is capable of answering questions about this web [page](http://pos.fflch.usp.br/).

yeah yeah yeah I know it is in portuguese, buuut soon I'll add examples in english.

---
# Test it

After installing all requirements...

You can run it and try it.

```
$ python bot.py
```

# Models evaluation

```
$ metrics bot.py
```
