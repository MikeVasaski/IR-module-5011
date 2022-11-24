from sklearn.feature_extraction.text import CountVectorizer
from nltk import PorterStemmer, word_tokenize
from nltk.corpus import stopwords
import m1
from scipy import sparse

def preProcess(s):
    ps = PorterStemmer()
    s = word_tokenize(s)
    stopwords_set = set(stopwords.words())
    s = [w for w in s if w not in stopwords_set]
    #s = [w for w in s if not w.isdigit()]
    s = [ps.stem(w) for w in s]
    s = ' '.join(s)
    return s

def sk_vectorize():
    cleaned_description = m1.get_and_clean_data()
    vectorizer = CountVectorizer(preprocessor=preProcess)
    vectorizer.fit(cleaned_description)
    query = vectorizer.transform(['good at java and python'])
    print(query)
    print(vectorizer.inverse_transform(query))

sk_vectorize()