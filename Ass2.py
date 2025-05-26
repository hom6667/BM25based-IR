# ass2.py (refactored for 2025 semester)
import os
import string
import numpy as np
import pandas as pd
from stemming.porter2 import stem

# Load stopwords list
def load_stopwords(filepath='common-english-words.txt'):
    with open(filepath, 'r') as file:
        return file.read().split(',')

# Load query information (the50Queries.txt)
def load_queries(filepath):
    datasets, titles, descs, narrs = [], [], [], []
    desc_accum, narr_accum = '', ''
    in_desc, in_narr = False, False

    with open(filepath, 'r') as file:
        for line in file:
            if '<desc>' in line:
                in_desc = True
                continue
            if in_desc and line.strip() == '':
                in_desc = False
                descs.append(desc_accum.strip())
                desc_accum = ''
                continue
            if in_desc:
                desc_accum += line.strip() + ' '
                continue

            if '<narr>' in line:
                in_narr = True
                continue
            if in_narr and line.strip() == '':
                in_narr = False
                narrs.append(narr_accum.strip())
                narr_accum = ''
                continue
            if in_narr:
                narr_accum += line.strip() + ' '
                continue

            if '<num>' in line:
                datasets.append(line.replace('<num> Number: R', '').strip())
                continue
            if '<title>' in line:
                titles.append(line.replace('<title> ', '').strip())
                continue

    return pd.DataFrame({
        'dataset': datasets,
        'titles': titles,
        'descriptions': descs,
        'narratives': narrs
    })

# Load evaluation data (relevance judgments)
def load_feedback():
    folder_path = os.path.join(os.getcwd(), 'dataset', 'EvaluationBenchmark')
    topic_ids, doc_ids, relevance = [], [], []

    for file in os.listdir(folder_path):
        with open(os.path.join(folder_path, file), 'r') as f:
            for line in f:
                tokens = line.strip().split()
                if len(tokens) == 3:
                    topic_ids.append(tokens[0])
                    doc_ids.append(tokens[1])
                    relevance.append(int(tokens[2]))

    return pd.DataFrame({
        'topic': topic_ids,
        'docid': doc_ids,
        'actual_rel': relevance
    })

# Class representing a single document
class Document:
    def __init__(self, doc_id):
        self.doc_id = doc_id
        self.terms = {}       # term: freq dictionary
        self.length = 0       # number of words in document

    def add_terms(self, term_freq):
        for term, count in term_freq.items():
            self.terms[term] = self.terms.get(term, 0) + count

    def __repr__(self):
        return f"<Document {self.doc_id}: {len(self.terms)} terms, {self.length} words>"

# Document collection
class Collection:
    def __init__(self):
        self.documents = []

    def add_document(self, document):
        self.documents.append(document)

    def get_doc_count(self):
        return len(self.documents)

    def get_doc_frequencies(self):
        df = {}
        for doc in self.documents:
            for term in doc.terms.keys():
                df[term] = df.get(term, 0) + 1
        return df

# Document parsing (BoW generation)
def parse_documents(path, stopwords):
    collection = Collection()
    for file in os.listdir(path):
        doc = Document(doc_id='')
        in_text_block = False
        word_count = 0

        with open(os.path.join(path, file), 'r') as f:
            for line in f:
                if 'itemid' in line:
                    doc.doc_id = ''.join(filter(str.isdigit, line))
                    continue
                if '<text>' in line:
                    in_text_block = True
                    continue
                if '</text>' in line:
                    break
                if in_text_block:
                    wc, tf = tokenize_line(line, stopwords)
                    word_count += wc
                    doc.add_terms(tf)

        doc.length = word_count
        collection.add_document(doc)
    return collection

# Text preprocessing and BoW generation
def tokenize_line(text, stopwords):
    strip_items = ['<p>', '</p>', '&quot;']
    for item in strip_items:
        text = text.replace(item, '')
    text = text.translate(str.maketrans('', '', string.digits))
    text = text.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
    text = text.lower().strip()

    words = text.split()
    valid_words = [w for w in words if len(w) >= 3]
    tokens = [stem(w) for w in valid_words if w not in stopwords]

    term_freq = {}
    for token in tokens:
        term_freq[token] = term_freq.get(token, 0) + 1

    return len(words), term_freq

# Calculate document frequency (for IDF)
def calculate_document_frequency(collection, verbose=False):
    df = collection.get_doc_frequencies()
    if verbose:
        print(f"Document Frequency (total terms: {len(df)}):")
        for term, count in sorted(df.items(), key=lambda x: x[1], reverse=True):
            print(f"{term}: {count}")
    return df


########################################################
# For task1 BM25IR
########################################################

def compute_bm25(collection, query_terms, doc_freqs, k1=1.2, k2=500, b=0.75):
    N = collection.get_doc_count()
    avg_doc_len = sum(doc.length for doc in collection.documents) / N
    scores = {}

    for doc in collection.documents:
        score = 0
        dl = doc.length
        K = k1 * ((1 - b) + b * (dl / avg_doc_len))

        for term, qf in query_terms.items():
            fi = doc.terms.get(term, 0)
            ni = doc_freqs.get(term, 0)

            if ni == 0 or fi == 0:
                continue  # Skip terms not in doc or collection

            idf = np.log10(1 + (N - ni + 0.5) / (ni + 0.5))
            term_weight = idf * ((k1 + 1) * fi) / (K + fi) * ((k2 + 1) * qf) / (k2 + qf)
            score += term_weight

        scores[doc.doc_id] = score

    # Sort in descending order
    ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked_docs

# Main Task1 execution function
def run_bm25ir(query_df, stopwords, dataset_base_path, output_dir, top_n=15):
    os.makedirs(output_dir, exist_ok=True)

    for idx, row in query_df.iterrows():
        dataset_id = row['dataset']
        dataset_path = os.path.join(dataset_base_path, f'Dataset{dataset_id}')
        collection = parse_documents(dataset_path, stopwords)

        # Query tokenization (based on title)
        _, query_terms = tokenize_line(row['titles'], stopwords)

        # Calculate document frequency
        df = calculate_document_frequency(collection, verbose=False)

        # Calculate BM25
        ranked_docs = compute_bm25(collection, query_terms, df)

        # Save top_n results
        output_path = os.path.join(output_dir, f'BM25_R{dataset_id}Ranking.dat')
        with open(output_path, 'w') as f:
            for doc_id, score in ranked_docs[:top_n]:
                f.write(f"{doc_id} {score}\n")

        print(f"BM25IR completed for Query R{dataset_id} - Top {top_n} saved to {output_path}")

########################################################
# For task2 # JM_LM (Task 2) - Jelinek-Mercer Language Model
########################################################

def compute_jm_lm(collection, query_terms, collection_df, lambda_=0.4):
    scores = {}

    # Total words in collection
    total_terms_in_coll = sum(
        sum(doc.terms.values()) for doc in collection.documents
    )

    for doc in collection.documents:
        doc_score = 0
        doc_term_count = sum(doc.terms.values())

        for term in query_terms.keys():
            f_qi_D = doc.terms.get(term, 0)
            c_qi = collection_df.get(term, 0)

            # JM smoothing formula
            p_doc = (1 - lambda_) * (f_qi_D / doc_term_count) if doc_term_count > 0 else 0
            p_coll = lambda_ * (c_qi / total_terms_in_coll) if total_terms_in_coll > 0 else 0

            # Accumulate log probability (using log10)
            if (p_doc + p_coll) > 0:
                doc_score += np.log10(p_doc + p_coll)

        scores[doc.doc_id] = doc_score

    # Sort in descending order
    ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked_docs

# Main Task 2 execution function
def run_jm_lm(query_df, stopwords, dataset_base_path, output_dir, top_n=15):
    os.makedirs(output_dir, exist_ok=True)

    for idx, row in query_df.iterrows():
        dataset_id = row['dataset']
        dataset_path = os.path.join(dataset_base_path, f'Dataset{dataset_id}')
        collection = parse_documents(dataset_path, stopwords)

        # Query tokenization (based on title)
        _, query_terms = tokenize_line(row['titles'], stopwords)

        # Calculate collection term frequencies
        collection_df = {}
        for doc in collection.documents:
            for term, freq in doc.terms.items():
                collection_df[term] = collection_df.get(term, 0) + freq

        # Calculate JM_LM
        ranked_docs = compute_jm_lm(collection, query_terms, collection_df)

        # Save top_n results
        output_path = os.path.join(output_dir, f'JM_LM_R{dataset_id}Ranking.dat')
        with open(output_path, 'w') as f:
            for doc_id, score in ranked_docs[:top_n]:
                f.write(f"{doc_id} {score}\n")

        print(f"JM_LM completed for Query R{dataset_id} - Top {top_n} saved to {output_path}")

########################################################
# For task3 Pseudo-Relevance Feedback
########################################################

def compute_prrm(collection, pseudo_query, doc_freqs, k1=1.2, k2=500, b=0.75):
    N = collection.get_doc_count()
    avg_doc_len = sum(doc.length for doc in collection.documents) / N
    scores = {}

    for doc in collection.documents:
        score = 0
        dl = doc.length
        K = k1 * ((1 - b) + b * (dl / avg_doc_len))

        for term, qf in pseudo_query.items():
            fi = doc.terms.get(term, 0)
            ni = doc_freqs.get(term, 0)

            if ni == 0 or fi == 0:
                continue

            idf = np.log10(1 + (N - ni + 0.5) / (ni + 0.5))
            term_score = idf * ((k1 + 1) * fi) / (K + fi) * ((k2 + 1) * qf) / (k2 + qf)
            score += term_score

        scores[doc.doc_id] = score

    ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked_docs

def run_prrm(query_df, stopwords, dataset_base_path, output_dir, top_n=15, pseudo_top=10):
    os.makedirs(output_dir, exist_ok=True)

    for idx, row in query_df.iterrows():
        dataset_id = row['dataset']
        dataset_path = os.path.join(dataset_base_path, f'Dataset{dataset_id}')
        collection = parse_documents(dataset_path, stopwords)

        # Create query BoW
        _, query_terms = tokenize_line(row['titles'], stopwords)
        doc_freqs = calculate_document_frequency(collection, verbose=False)

        # Perform initial BM25IR
        initial_ranked = compute_bm25(collection, query_terms, doc_freqs)

        # Calculate word frequencies from pseudo-relevant documents
        pseudo_query = {}
        for doc_id, _ in initial_ranked[:pseudo_top]:
            doc = next((d for d in collection.documents if d.doc_id == doc_id), None)
            if doc:
                for term, freq in doc.terms.items():
                    pseudo_query[term] = pseudo_query.get(term, 0) + freq

        # Calculate PRRM scores
        final_ranked = compute_prrm(collection, pseudo_query, doc_freqs)

        # Save top_n results
        output_path = os.path.join(output_dir, f'My_PRRM_R{dataset_id}Ranking.dat')
        with open(output_path, 'w') as f:
            for doc_id, score in final_ranked[:top_n]:
                f.write(f"{doc_id} {score}\n")

        print(f"PRRM completed for Query R{dataset_id} - Top {top_n} saved to {output_path}")