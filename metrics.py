from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from numpy import mean

# Load training data

with open("base_treino_secretaria.csv", encoding='utf-8') as f:
    sentencas = f.read().split("\n")

label = []
interacao = []

for sentenca in sentencas[:-1]:
    sentenca = sentenca.split(";")
    label.append(sentenca[0])
    interacao.append(sentenca[1])

# Load stopwords

with open("stop_words.csv", encoding="utf-8") as f:
    pt_stop_words = f.read().split("\n")

####Tratando as features da base de treino####

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words=pt_stop_words)
features = tfidf.fit_transform(interacao).toarray()
labels = label

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(interacao)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

#### Instanciando modelos####
NB = MultinomialNB().fit(X_train_tfidf, labels)
#prcpt = Perceptron().fit(X_train_tfidf, labels)
SVM = svm.SVC(kernel='linear', C=0.1).fit(X_train_tfidf, labels)
log = LogisticRegression().fit(X_train_tfidf, labels)

####Avaliando cada Classificador####
from NB import naive_bayes
with open("teste_maior.csv", encoding="utf-8") as f:
    test_dados = f.read().split("\n")

    x_test = []
    y_test = []
    for linha in test_dados[:-1]:
        linha = linha.split(';')
        x_test.append(linha[1])
        y_test.append(linha[0])

y_pred_SVM = []
y_pred_log = []
y_pred_NB_sk = []
y_pred_NB = []
for x,y in zip(x_test,y_test):
    prediction = (SVM.predict(count_vect.transform([x])))
    y_pred_SVM.append(prediction[0])
    prediction = (log.predict(count_vect.transform([x])))
    y_pred_log.append(prediction[0])
    prediction = (NB.predict(count_vect.transform([x])))
    y_pred_NB_sk.append(prediction[0])
    prediction = (naive_bayes(x))
    y_pred_NB.append(prediction)

df_SVM = pd.DataFrame()
df_SVM['true'] = y_test
df_SVM['predicted'] = y_pred_SVM
df_log = pd.DataFrame()
df_log['true'] = y_test
df_log['predicted'] = y_pred_log
df_NB_sk = pd.DataFrame()
df_NB_sk['true'] = y_test
df_NB_sk['predicted'] = y_pred_NB_sk
df_NB = pd.DataFrame()
df_NB['true'] = y_test
df_NB['predicted'] = y_pred_NB

#Entrada
# tabela : tabela é um data.frame com strings e o array escroto com colunas com os nomes "predicted" e "true"

def metricas(tabela):
    #replace dos dados
    tabela["predicted"] = tabela["predicted"].apply(lambda x : x.replace("\'",""))
    # pega todos os labels
    labels = tabela.true.unique()
    labels
    precisions = []
    recalls = []
    accuracies = []
    # calcula as métricas para cada label
    for label in labels:
        # binarize
        true_bin = tabela["true"] == label
        predicted_bin = tabela["predicted"] == label
        # calculate metrics
        precisions.append(precision_score(y_true=true_bin, y_pred=predicted_bin))
        recalls.append(recall_score(y_true=true_bin, y_pred=predicted_bin))
        accuracies.append(accuracy_score(y_true=true_bin, y_pred=predicted_bin))
    #print(precisions)
    print("Precisions average: ", mean(precisions))
    print("\nRecalls average: ", mean(recalls))
    print("\nAccuracies average:", mean(accuracies))
    print("------------------------------------------")

for nome, classificador in zip(['SVM:\n','Naive Bayes:\n','Logistic Regression:\n','Naive Bayes (scikit-learn):\n'],[df_SVM, df_NB, df_log, df_NB_sk]):
    print(nome)
    metricas(classificador)
