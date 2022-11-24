import string
import nltk
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


def get_and_clean_data():
    data = pd.read_csv('software_developer_united_states_1971_20191023_1.csv')
    description = data['job_description']
    cleaned_description = description.apply(lambda s: s.translate(str.maketrans('', '', string.punctuation + u'\xa0')))
    cleaned_description = cleaned_description.apply(lambda s: s.lower())
    cleaned_description = cleaned_description.apply(lambda s: s.translate(str.maketrans(string.whitespace, ' ' * len(string.whitespace), '')))
    cleaned_description = cleaned_description.drop_duplicates()
    return cleaned_description


def simple_tokenize(data):
    cleaned_description = data.apply(lambda s: [x.strip() for x in s.split()])
    return cleaned_description


def parse_job_description():
    cleaned_description = get_and_clean_data()
    cleaned_description = simple_tokenize(cleaned_description)
    return cleaned_description


def count_python_mysql():
    parsed_description = parse_job_description()
    count_python = parsed_description.apply(lambda s: 'python' in s).sum()
    count_mysql = parsed_description.apply(lambda s: 'mysql' in s).sum()
    count_java = parsed_description.apply(lambda s: 'java' in s).sum()
    print('python: ' + str(count_python) + ' of ' + str(parsed_description.shape[0]))
    print('mysql: ' + str(count_mysql) + ' of ' + str(parsed_description.shape[0]))
    print('java: ' + str(count_java) + ' of ' + str(parsed_description.shape[0]))


def parse_db():
    html_doc = requests.get("https://db-engines.com/en/ranking").content
    soup = BeautifulSoup(html_doc, 'html.parser')
    db_table = soup.find("table", {"class": "dbi"})
    all_db = [''.join(s.find('a').findAll(text=True, recursive=False)).strip() for s in db_table.findAll("th", {"class": "pad-l"})]
    all_db = list(dict.fromkeys(all_db))
    db_list = all_db[:10]
    db_list = [s.lower() for s in db_list]
    db_list = [[x.strip() for x in s.split()] for s in db_list]
    return db_list


def parsed_des():
    cleaned_db = parse_db()
    parsed_description = parse_job_description()
    raw = [None] * len(cleaned_db)

    for i,db in enumerate(cleaned_db):
        raw[i] = parsed_description.apply(lambda s: np.all([x in s for x in db])).sum()
        print(' '.join(db) + ': ' + str(raw[i]) + ' of ' + str(parsed_description.shape[0]))
    # python
    with_python = [None] * len(cleaned_db)
    for i,db in enumerate(cleaned_db):
        with_python[i] = parsed_description.apply(lambda s: np.all([x in s for x in db]) and 'python' in s).sum()
        print(' '.join(db) + ' + python: ' + str(with_python[i]) + ' of ' + str(parsed_description.shape[0]))
    for i, db in enumerate(cleaned_db):
        print(' '.join(db) + ' + python: ' + str(with_python[i]) + ' of ' + str(raw[i]) + ' (' + str(np.around(with_python[i] / raw[i] * 100, 2)) + '%)')
    # -
    # java
    with_java = [None] * len(cleaned_db)
    for i,db in enumerate(cleaned_db):
        with_java[i] = parsed_description.apply(lambda s: np.all([x in s for x in db]) and 'java' in s).sum()
        print(' '.join(db) + ' + java: ' + str(with_java[i]) + ' of ' + str(parsed_description.shape[0]))
    for i, db in enumerate(cleaned_db):
        print(' '.join(db) + ' + java: ' + str(with_java[i]) + ' of ' + str(raw[i]) + ' (' + str(np.around(with_java[i] / raw[i] * 100, 2)) + '%)')
    # -


def create_index():
    lang = [['java'],['python'],['c'],['kotlin'],['swift'],['rust'],['ruby'],['scala'],['julia'], ['lua']]
    parsed_description = parse_job_description()
    parsed_db = parse_db()
    all_terms = lang + parsed_db
    query_map = pd.DataFrame(parsed_description.apply(lambda s: [1 if np.all([d in s for d in db]) else 0 for db in all_terms]).values.tolist(), columns=[' '.join(d) for d in all_terms])
    return query_map


nltk.download('stopwords')
nltk.download('punkt')


def tokenizer():
    str1 = 'the chosen software developer will be part of a larger engineering team developing software for medical devices.'
    str2 = 'we are seeking a seasoned software developer with strong analytical and technical skills to join our public sector technology consulting team.'
    tokened_str1 = word_tokenize(str1)
    tokened_str2 = word_tokenize(str2)
    tokened_str1 = [w for w in tokened_str1 if len(w) > 2]
    tokened_str2 = [w for w in tokened_str2 if len(w) > 2]

    no_sw_str1 = [word for word in tokened_str1 if not word in stopwords.words()]
    no_sw_str2 = [word for word in tokened_str2 if not word in stopwords.words()]

    ps = PorterStemmer()
    stemmed_str1 = np.unique([ps.stem(w) for w in no_sw_str1])
    stemmed_str2 = np.unique([ps.stem(w) for w in no_sw_str2])
    full_list = np.sort(np.concatenate([stemmed_str1, stemmed_str2]))
    return full_list