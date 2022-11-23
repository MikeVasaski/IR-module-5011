import nltk
import numpy as np

import m1

nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


def inverse_indexing(parsed_description):
    sw_set = set(stopwords.words()) - {'c'}
    no_sw_description = parsed_description.apply(lambda x: [w for w in x if w not in sw_set])
    ps = PorterStemmer()
    stemmed_description = no_sw_description.apply(lambda x: set([ps.stem(w) for w in x]))
    all_unique_term = list(set.union(*stemmed_description.to_list()))
    invert_idx = {}
    for s in all_unique_term:
        invert_idx[s] = set(stemmed_description.loc[stemmed_description.apply(lambda x: s in x)].index)
    return invert_idx


def search(invert_idx, query):
    ps = PorterStemmer()
    processed_query = [s.lower() for s in query.split()]
    stemmed = [ps.stem(s) for s in processed_query]
    matched = set.intersection(*[invert_idx[s] for s in stemmed])
    return matched


if __name__ == '__main__':
    parsed_description = m1.parse_job_description()
    invert_idx = inverse_indexing(parsed_description)
    query = 'java oracle'
    matched = search(invert_idx, query)
    print(parsed_description.loc[matched].apply(lambda x: ' '.join(x)).head().to_markdown())