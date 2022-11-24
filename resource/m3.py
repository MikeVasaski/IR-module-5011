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
    vectorizer = CountVectorizer(preprocessor=preProcess,ngram_range=(1, 2))
    vectorizer.fit(cleaned_description)
    query = vectorizer.transform(['good at java and python'])
    print('Query >> '+query)
    print(vectorizer.inverse_transform(query))

    query = vectorizer.transform(['good at java and python'])
    print('Query >> '+query)
    print(vectorizer.inverse_transform(query))

    query = vectorizer.transform(['good at python and java'])
    print('Query >> '+query)
    print(vectorizer.inverse_transform(query))
    print(vectorizer.get_feature_names())

    #tri_gram_vectorizer.inverse_transform(tri_gram_vectorizer.transform([cleaned_description[0]]))

sk_vectorize()