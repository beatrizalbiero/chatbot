from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


#####Load Train Data#####
with open("base_treino_secretaria.csv", encoding='utf-8') as f:
    sentencas = f.read().split("\n")

label = []
interacao = []

for sentenca in sentencas[:-1]:
    sentenca = sentenca.split(";")
    label.append(sentenca[0])
    interacao.append(sentenca[1])


####Load Stop Words#####
with open("stop_words.csv", encoding="utf-8") as f:
    pt_stop_words = f.read().split("\n")

####Features####

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words = pt_stop_words)
features = tfidf.fit_transform(interacao).toarray()
labels = label

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(interacao)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

###Model###

log = LogisticRegression().fit(X_train_tfidf, labels)

def classification(interaction):
    """
    Classify an interaction.

    :interaction type: str
    :rtype: list
    """
    return log.predict(count_vect.transform([interaction]))[0]
