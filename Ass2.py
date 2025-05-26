# ass2.py

import os
import string
import numpy as np
import pandas as pd
from stemming.porter2 import stem

# Load stopwords
def load_stopwords(filepath='common-english-words.txt'):
    with open(filepath, 'r') as f:
        stopwords = f.read().split(',')
    return stopwords

# Load queries from the50Queries.txt
def load_queries(filepath):
    datasets, titles, descs, narrs = [], [], [], []
    desc_text, narr_text = '', ''
    flag_desc, flag_narr = 0, 0

    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('<desc>'):
                flag_desc = 1
                continue
            if flag_desc and len(line.strip()) == 0:
                flag_desc = 0
                descs.append(desc_text.strip())
                desc_text = ''
                continue
            if flag_desc:
                desc_text += line.strip() + ' '
                continue

            if line.startswith('<narr>'):
                flag_narr = 1
                continue
            if flag_narr and len(line.strip()) == 0:
                flag_narr = 0
                narrs.append(narr_text.strip())
                narr_text = ''
                continue
            if flag_narr:
                narr_text += line.strip() + ' '
                continue

            if line.startswith('<num>'):
                datasets.append(line.strip('<num> Number: R').strip())
                continue
            if line.startswith('<title>'):
                titles.append(line.strip('<title> ').strip())
                continue

    return pd.DataFrame({
        'dataset': datasets,
        'titles': titles,
        'descriptions': descs,
        'narratives': narrs
    })

# For task5 i will code the following function
def load_feedback():
    folderpath_feedback = os.getcwd() + r'\dataset\EvaluationBenchmark'
    list_topic = []
    list_docid = [] 
    list_rel = []

    for file in os.listdir(folderpath_feedback):
        with open(os.path.join(folderpath_feedback, file), 'r') as f:
            for row in f.read().split('\n'):
                try:
                    l = row.split()
                    list_topic.append(l[0])
                    list_docid.append(l[1])
                    list_rel.append(l[2])
                except:
                    pass # (빈 줄 등 예외 처리)

    return pd.DataFrame({
        'topic': list_topic,
        'docid': list_docid,
        'actual_rel': [int(val) for val in list_rel]
    })

class DocWords:
    def __init__(self, docID, terms, doc_len):
        self.docID = docID          # 문서 ID
        self.terms = terms          # {'term1': freq1, 'term2': freq2, ...}
        self.doc_len = doc_len      # 문서의 단어 수 (길이)

    def set_doc_len(self, count):
        self.doc_len = count

    def getDocId(self):
        return self.docID

    def getDocLen(self):
        return self.doc_len

    def get_term_list(self):
        # 단어-빈도 딕셔너리를 빈도수 기준으로 정렬
        return dict(sorted(self.terms.items(), key=lambda x:x[1], reverse=True))

    def addTerm(self, tf_dict):
        # 새로운 term-freq 딕셔너리를 합쳐줌 (이미 있으면 더함)
        for key in tf_dict:
            try:
                self.terms[key] += tf_dict[key]
            except:
                self.terms[key] = tf_dict[key]

class BowColl:
    def __init__(self, Coll, weights):
        self.Coll = Coll          # DocWords 객체들의 리스트 (컬렉션)
        self.weights = weights    # 각 문서별 단어 가중치 딕셔너리 리스트
        self.numOfDocs = 0        # 문서 수

    def addDocWords(self, DocWords):
        # DocWords 객체를 컬렉션에 추가
        self.Coll.append(DocWords)
        self.numOfDocs += 1

    def addWeights(self, weights):
        # 가중치 딕셔너리 추가 (보통 쿼리 확장할 때 사용 가능)
        self.weights.append(weights)

    def getNumOfDocs(self):
        return self.numOfDocs

    def getSummary(self):
        # 컬렉션에 들어있는 각 문서 요약 출력 (문서ID, term 수, 단어 수 등)
        for d in self.Coll:
            print(f"Document {d.docID}: {len(d.terms)} terms, {d.doc_len} words")
def parse_docs(inputpath, stop_words, include_files=[]):
    # inputpath: 데이터셋 폴더 경로
    # stop_words: 불용어 리스트
    # include_files: 특정 파일만 처리할 때 (비워두면 전체)

    bow_coll = BowColl([], [])  # 새로운 BowColl 초기화

    for file_path in os.listdir(inputpath):
        if len(include_files) == 0 or (file_path in include_files):
            curr_doc = DocWords(docID='', terms={}, doc_len=0)
            word_count = 0
            flag = 0
            line_doc = {}

            with open(os.path.join(inputpath, file_path), 'r') as f:
                for line in f:
                    # 문서 ID 읽기
                    if 'itemid' in line:
                        bits = line.split()
                        for b in bits:
                            if b.startswith('itemid'):
                                b = b.translate(str.maketrans('', '', string.ascii_letters)) \
                                     .translate(str.maketrans('', '', string.punctuation))
                                curr_doc.docID = b
                                break

                    # <text> ~ </text> 사이만 처리
                    if line.startswith('<text>'):
                        flag = 1
                        continue
                    if line.startswith('</text>'):
                        break
                    if flag == 1:
                        line_word_count, line_doc = parse_query(line, stop_words)
                        word_count += line_word_count
                        curr_doc.addTerm(line_doc)

            curr_doc.doc_len = word_count  # 문서 길이 저장
            bow_coll.addDocWords(curr_doc) # 컬렉션에 추가

    return bow_coll