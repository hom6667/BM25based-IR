"""
Information Retrieval Models Implementation (IFN647 Assignment 2)

This program implements and evaluates three information retrieval models:
1. BM25IR (Best Match 25 Information Retrieval)
2. LMRM (Language Model with Relevance Model)
3. PRRM (Pseudo-Relevance Feedback with Relevance Model)

The program processes 50 datasets/queries and evaluates the models using:
- Mean Average Precision (MAP)
- Precision@12
- DCG@12 (Discounted Cumulative Gain at 12)
"""

import os
import string
import numpy as np
import pandas as pd
from stemming.porter2 import stem
import re

# Load stopwords list
def load_stopwords(filepath='dataset/common-english-words.txt'):
    """
    Loads a list of common English stopwords from a file.
    Stopwords are common words that are typically filtered out before processing text
    as they don't carry significant meaning (e.g., 'the', 'is', 'at').
    """
    with open(filepath, 'r') as file:
        return file.read().split(',')

def load_queries(filepath):
    """
    Loads and parses query information from an XML-like format file.
    Each query contains:
    - dataset: Query identifier (R101-R150)
    - title: The main query text
    - description: Detailed description of the information need
    - narrative: Explanation of what makes a document relevant
    """
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
    """
    Loads relevance judgments from evaluation benchmark files.
    Each line contains:
    - topic_id: Query identifier
    - doc_id: Document identifier
    - relevance: Binary relevance score (1 for relevant, 0 for non-relevant)
    """
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
    """
    Represents a single document in the collection.
    Attributes:
    - doc_id: Unique identifier for the document
    - terms: Dictionary mapping terms to their frequencies in the document
    - length: Total number of words in the document
    """
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
    """
    Represents a collection of documents.
    Provides methods to:
    - Add documents to the collection
    - Get the total number of documents
    - Calculate document frequencies for terms
    """
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
    """
    Parses XML documents and creates a document collection.
    For each document:
    1. Extracts the document ID from the newsitem tag
    2. Processes the text content
    3. Creates a bag-of-words representation
    4. Calculates document length
    """
    collection = Collection()
    for file in os.listdir(path):
        doc = Document(doc_id='')
        in_text_block = False
        word_count = 0

        with open(os.path.join(path, file), 'r') as f:
            for line in f:
                # Extract itemid from the newsitem tag attribute
                if '<newsitem' in line and 'itemid="' in line:
                    match = re.search(r'itemid="(\d+)"', line)
                    if match:
                        doc.doc_id = match.group(1)
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
    """
    Preprocesses text and generates a bag-of-words representation.
    Steps:
    1. Removes HTML tags and special characters
    2. Converts to lowercase
    3. Removes numbers
    4. Removes punctuation
    5. Removes stopwords
    6. Stems words
    7. Counts term frequencies
    """
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
    """
    Calculates document frequency (DF) for each term in the collection.
    DF is the number of documents containing each term.
    Used in IDF calculation for BM25IR.
    """
    df = collection.get_doc_frequencies()
    if verbose:
        print(f"Document Frequency (total terms: {len(df)}):")
        for term, count in sorted(df.items(), key=lambda x: x[1], reverse=True):
            print(f"{term}: {count}")
    return df


########################################################
# Task 1: BM25IR Implementation
########################################################

def compute_bm25(collection, query_terms, doc_freqs, k1=1.2, k2=500, b=0.75):
    """
    Implements the BM25 ranking function.
    BM25 is a bag-of-words retrieval function that ranks documents based on:
    1. Term frequency in document (TF)
    2. Inverse document frequency (IDF)
    3. Document length normalization
    
    Parameters:
    - k1: Term frequency scaling parameter (default: 1.2)
    - k2: Query term frequency scaling parameter (default: 500)
    - b: Document length normalization parameter (default: 0.75)
    """
    N = collection.get_doc_count()  # N: total number of documents
    avg_doc_len = sum(doc.length for doc in collection.documents) / N  # avdl: average document length
    scores = {}

    for doc in collection.documents:
        score = 0
        dl = doc.length  # dl: document length
        K = k1 * ((1 - b) + b * (dl / avg_doc_len))  # K = k1*((1-b) + b*dl/avdl)

        for term, qf_qi in query_terms.items():
            f_qi_D = doc.terms.get(term, 0)  # f_qi,D: frequency of query term in document
            n_qi = doc_freqs.get(term, 0)  # n_qi: frequency of term in collection

            if n_qi == 0 or f_qi_D == 0:
                continue

            # IDF calculation exactly as specified (log2)
            idf = np.log2(1 + (N - n_qi + 0.5) / (n_qi + 0.5))
            
            # BM25 score calculation exactly as specified
            term_weight = idf * ((k1 + 1) * f_qi_D) / (K + f_qi_D) * ((k2 + 1) * qf_qi) / (k2 + qf_qi)
            score += term_weight

        scores[doc.doc_id] = score

    # Sort in descending order
    ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked_docs

def run_bm25ir(query_df, stopwords, dataset_base_path, output_dir, top_n=12):
    """
    Runs BM25IR for all queries and saves results.
    For each query:
    1. Loads the corresponding dataset
    2. Processes the query
    3. Calculates BM25 scores
    4. Saves top N results to file
    """
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
        output_path = os.path.join(output_dir, f'BM25IR_R{dataset_id}Ranking.dat')
        with open(output_path, 'w') as f:
            for doc_id, score in ranked_docs[:top_n]:
                f.write(f"{doc_id} {score:.15f}\n")  # Format score to match specification

        print(f"BM25IR completed for Query R{dataset_id} - Top {top_n} saved to {output_path}")

########################################################
# Task 2: LMRM Implementation (Jelinek-Mercer Language Model)
########################################################

def compute_lmrm(collection, query_terms, collection_df, lambda_=0.4):
    """
    Implements the Jelinek-Mercer Language Model with Relevance Model.
    Uses linear interpolation smoothing:
    P(t|D) = (1-λ)P(t|D) + λP(t|C)
    where:
    - P(t|D) is the probability of term t in document D
    - P(t|C) is the probability of term t in collection C
    - λ is the smoothing parameter (default: 0.4)
    """
    scores = {}
    C_size = sum(  # |C|: total number of words in collection
        sum(doc.terms.values()) for doc in collection.documents
    )

    for doc in collection.documents:
        doc_score = 0
        D_size = sum(doc.terms.values())  # |D|: total number of words in document

        for term in query_terms.keys():
            f_qi_D = doc.terms.get(term, 0)  # f_qi,D: frequency of query term in document
            c_qi = collection_df.get(term, 0)  # c_qi: frequency of term in collection

            # JM smoothing formula exactly as specified (log2)
            if D_size > 0 and C_size > 0:
                p_doc = (1 - lambda_) * (f_qi_D / D_size)
                p_coll = lambda_ * (c_qi / C_size)
                prob = p_doc + p_coll
                if prob > 0:
                    doc_score += np.log2(prob)

        scores[doc.doc_id] = doc_score

    # Sort in descending order
    ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked_docs

def run_lmrm(query_df, stopwords, dataset_base_path, output_dir, top_n=12):
    """
    Runs LMRM for all queries and saves results.
    For each query:
    1. Loads the corresponding dataset
    2. Processes the query
    3. Calculates collection term frequencies
    4. Applies LMRM scoring
    5. Saves top N results to file
    """
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

        # Calculate LMRM
        ranked_docs = compute_lmrm(collection, query_terms, collection_df)

        # Save top_n results
        output_path = os.path.join(output_dir, f'LMRM_R{dataset_id}Ranking.dat')
        with open(output_path, 'w') as f:
            for doc_id, score in ranked_docs[:top_n]:
                f.write(f"{doc_id} {score:.15f}\n")  # Format score to match specification

        print(f"LMRM completed for Query R{dataset_id} - Top {top_n} saved to {output_path}")

########################################################
# Task 3: PRRM Implementation (Pseudo-Relevance Feedback)
########################################################

def compute_prrm(collection, pseudo_query, doc_freqs, k1=1.2, k2=500, b=0.75):
    """
    Implements Pseudo-Relevance Feedback with Relevance Model.
    Process:
    1. Uses initial BM25IR results to identify pseudo-relevant documents
    2. Expands query using terms from pseudo-relevant documents
    3. Reranks documents using expanded query
    """
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

def run_prrm(query_df, stopwords, dataset_base_path, output_dir, top_n=12, pseudo_top=10):
    """
    Runs PRRM for all queries and saves results.
    For each query:
    1. Performs initial BM25IR ranking
    2. Selects top pseudo_top documents as pseudo-relevant
    3. Creates expanded query from pseudo-relevant documents
    4. Reranks using expanded query
    5. Saves top N results to file
    """
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
        expanded_query = {}  # query expanded from pseudo-relevant documents
        for doc_id, _ in initial_ranked[:pseudo_top]:
            doc = next((d for d in collection.documents if d.doc_id == doc_id), None)
            if doc:
                for term, freq in doc.terms.items():
                    expanded_query[term] = expanded_query.get(term, 0) + freq

        # Calculate PRRM scores
        final_ranked = compute_prrm(collection, expanded_query, doc_freqs)

        # Save top_n results
        output_path = os.path.join(output_dir, f'PRRM_R{dataset_id}Ranking.dat')
        with open(output_path, 'w') as f:
            for doc_id, score in final_ranked[:top_n]:
                f.write(f"{doc_id} {score:.15f}\n")  # Format score to match specification

        print(f"PRRM completed for Query R{dataset_id} - Top {top_n} saved to {output_path}")

########################################################
# Task 5: Evaluation Metrics Implementation
########################################################

def calculate_ap(predicted, relevant):
    """
    Calculates Average Precision (AP).
    AP is the mean of the precision values after each relevant document is retrieved.
    """
    hits, sum_precisions = 0, 0.0
    for i, doc_id in enumerate(predicted):
        if doc_id in relevant:
            hits += 1
            sum_precisions += hits / (i + 1)
    return sum_precisions / len(relevant) if relevant else 0.0

def calculate_precision_at_k(predicted, relevant, k=12):
    """
    Calculates Precision@k.
    Precision@k is the proportion of relevant documents in the top k results.
    """
    hits = sum(1 for doc_id in predicted[:k] if doc_id in relevant)
    return hits / k

def calculate_dcg_at_k(predicted, relevant, k=12):
    """
    Calculates Discounted Cumulative Gain at k (DCG@k).
    DCG@k considers both the relevance and the position of documents in the ranking.
    """
    dcg = 0.0
    for i, doc_id in enumerate(predicted[:k]):
        if doc_id in relevant:
            dcg += 1 / np.log2(i + 2)  # log2(i+2) because i starts from 0
    return dcg

def evaluate_model(predicted_file, relevance_file, k=12):
    """
    Evaluates a model's performance using AP, P@k, and DCG@k.
    Loads predicted rankings and relevance judgments, then calculates metrics.
    """
    # Load predicted ranked documents
    with open(predicted_file, 'r') as f:
        predicted = [line.strip().split()[0] for line in f if line.strip()]
    
    # Load relevant documents
    with open(relevance_file, 'r') as f:
        relevant = [line.strip().split()[1] for line in f if line.strip().split()[2] == '1']

    ap = calculate_ap(predicted, relevant)
    p_at_k = calculate_precision_at_k(predicted, relevant, k)
    dcg = calculate_dcg_at_k(predicted, relevant, k)

    return ap, p_at_k, dcg

def run_task5_evaluation(output_dir='RankingOutputs', relevance_dir='EvaluationBenchmark', k=12):
    """
    Runs evaluation for all models and queries.
    Generates three tables:
    1. Average Precision results
    2. Precision@12 results
    3. DCG@12 results
    
    Each table shows per-query results and averages for each model.
    """
    models = ['BM25IR', 'LMRM', 'PRRM']
    results = {model: {'AP': [], 'P@12': [], 'DCG@12': []} for model in models}

    for model in models:
        for i in range(101, 151):
            pred_file = os.path.join(output_dir, f'{model}_R{i}Ranking.dat')
            rel_file = os.path.join(relevance_dir, f'Dataset{i}.txt')
            ap, p12, dcg = evaluate_model(pred_file, rel_file, k)
            results[model]['AP'].append(ap)
            results[model]['P@12'].append(p12)
            results[model]['DCG@12'].append(dcg)

    # Table 1: Average Precision
    print("\nTable 1. The performance of 3 models on Average Precision")
    print(f"{'Topic':<6} {'BM25IR':<10} {'LMRM':<10} {'PRRM':<10}")
    for idx, i in enumerate(range(101, 151)):
        print(f"R{i:<4} {results['BM25IR']['AP'][idx]:<10.4f} {results['LMRM']['AP'][idx]:<10.4f} {results['PRRM']['AP'][idx]:<10.4f}")
    print(f"{'MAP':<6} {np.mean(results['BM25IR']['AP']):<10.4f} {np.mean(results['LMRM']['AP']):<10.4f} {np.mean(results['PRRM']['AP']):<10.4f}")

    # Table 2: Precision@12
    print("\nTable 2. The performance of 3 models on precision@12")
    print(f"{'Topic':<6} {'BM25IR':<10} {'LMRM':<10} {'PRRM':<10}")
    for idx, i in enumerate(range(101, 151)):
        print(f"R{i:<4} {results['BM25IR']['P@12'][idx]:<10.4f} {results['LMRM']['P@12'][idx]:<10.4f} {results['PRRM']['P@12'][idx]:<10.4f}")
    print(f"{'Average':<6} {np.mean(results['BM25IR']['P@12']):<10.4f} {np.mean(results['LMRM']['P@12']):<10.4f} {np.mean(results['PRRM']['P@12']):<10.4f}")

    # Table 3: DCG@12
    print("\nTable 3. The performance of 3 models on DCG12")
    print(f"{'Topic':<6} {'BM25IR':<10} {'LMRM':<10} {'PRRM':<10}")
    for idx, i in enumerate(range(101, 151)):
        print(f"R{i:<4} {results['BM25IR']['DCG@12'][idx]:<10.4f} {results['LMRM']['DCG@12'][idx]:<10.4f} {results['PRRM']['DCG@12'][idx]:<10.4f}")
    print(f"{'Average':<6} {np.mean(results['BM25IR']['DCG@12']):<10.4f} {np.mean(results['LMRM']['DCG@12']):<10.4f} {np.mean(results['PRRM']['DCG@12']):<10.4f}")

    return results

def run_task5_evaluation_top10(output_dir='RankingOutputs', relevance_dir='EvaluationBenchmark', k=12):
    """
    Runs evaluation for all models and queries, showing only top 10 results.
    Generates three tables showing only the top 10 performing queries for each metric.
    """
    models = ['BM25IR', 'LMRM', 'PRRM']
    results = {model: {'AP': [], 'P@12': [], 'DCG@12': []} for model in models}
    query_ids = list(range(101, 151))  # R101 to R150

    # Calculate results for all queries
    for model in models:
        for i in range(101, 151):
            pred_file = os.path.join(output_dir, f'{model}_R{i}Ranking.dat')
            rel_file = os.path.join(relevance_dir, f'Dataset{i}.txt')
            ap, p12, dcg = evaluate_model(pred_file, rel_file, k)
            results[model]['AP'].append(ap)
            results[model]['P@12'].append(p12)
            results[model]['DCG@12'].append(dcg)

    # Table 1: Top 10 Average Precision
    print("\nTable 1. Top 10 queries by Average Precision")
    print(f"{'Topic':<6} {'BM25IR':<10} {'LMRM':<10} {'PRRM':<10}")
    
    # Calculate average AP for each query across models
    query_ap_scores = []
    for i in range(len(query_ids)):
        avg_ap = np.mean([results[model]['AP'][i] for model in models])
        query_ap_scores.append((query_ids[i], avg_ap))
    
    # Sort by average AP and get top 10
    top10_ap = sorted(query_ap_scores, key=lambda x: x[1], reverse=True)[:10]
    
    for query_id, _ in top10_ap:
        idx = query_ids.index(query_id)
        print(f"R{query_id:<4} {results['BM25IR']['AP'][idx]:<10.4f} {results['LMRM']['AP'][idx]:<10.4f} {results['PRRM']['AP'][idx]:<10.4f}")
    print(f"{'MAP':<6} {np.mean(results['BM25IR']['AP']):<10.4f} {np.mean(results['LMRM']['AP']):<10.4f} {np.mean(results['PRRM']['AP']):<10.4f}")

    # Table 2: Top 10 Precision@12
    print("\nTable 2. Top 10 queries by Precision@12")
    print(f"{'Topic':<6} {'BM25IR':<10} {'LMRM':<10} {'PRRM':<10}")
    
    # Calculate average P@12 for each query across models
    query_p12_scores = []
    for i in range(len(query_ids)):
        avg_p12 = np.mean([results[model]['P@12'][i] for model in models])
        query_p12_scores.append((query_ids[i], avg_p12))
    
    # Sort by average P@12 and get top 10
    top10_p12 = sorted(query_p12_scores, key=lambda x: x[1], reverse=True)[:10]
    
    for query_id, _ in top10_p12:
        idx = query_ids.index(query_id)
        print(f"R{query_id:<4} {results['BM25IR']['P@12'][idx]:<10.4f} {results['LMRM']['P@12'][idx]:<10.4f} {results['PRRM']['P@12'][idx]:<10.4f}")
    print(f"{'Average':<6} {np.mean(results['BM25IR']['P@12']):<10.4f} {np.mean(results['LMRM']['P@12']):<10.4f} {np.mean(results['PRRM']['P@12']):<10.4f}")

    # Table 3: Top 10 DCG@12
    print("\nTable 3. Top 10 queries by DCG@12")
    print(f"{'Topic':<6} {'BM25IR':<10} {'LMRM':<10} {'PRRM':<10}")
    
    # Calculate average DCG@12 for each query across models
    query_dcg_scores = []
    for i in range(len(query_ids)):
        avg_dcg = np.mean([results[model]['DCG@12'][i] for model in models])
        query_dcg_scores.append((query_ids[i], avg_dcg))
    
    # Sort by average DCG@12 and get top 10
    top10_dcg = sorted(query_dcg_scores, key=lambda x: x[1], reverse=True)[:10]
    
    for query_id, _ in top10_dcg:
        idx = query_ids.index(query_id)
        print(f"R{query_id:<4} {results['BM25IR']['DCG@12'][idx]:<10.4f} {results['LMRM']['DCG@12'][idx]:<10.4f} {results['PRRM']['DCG@12'][idx]:<10.4f}")
    print(f"{'Average':<6} {np.mean(results['BM25IR']['DCG@12']):<10.4f} {np.mean(results['LMRM']['DCG@12']):<10.4f} {np.mean(results['PRRM']['DCG@12']):<10.4f}")

    return results

########################################################
# Main Execution
########################################################

if __name__ == "__main__":
    """
    Main execution flow:
    1. Load stopwords and queries
    2. Run BM25IR for all queries
    3. Run LMRM for all queries
    4. Run PRRM for all queries
    5. Evaluate all models and print results
    6. Print top 10 results for each metric
    """
    # Load stopwords and queries
    stopwords = load_stopwords('common-english-words.txt')
    query_df = load_queries('Queries-1.txt')

    dataset_base_path = 'dataset'
    output_dir = 'RankingOutputs'

    # Run BM25IR
    run_bm25ir(query_df, stopwords, dataset_base_path, output_dir, top_n=12)

    # Run LMRM
    run_lmrm(query_df, stopwords, dataset_base_path, output_dir, top_n=12)

    # Run PRRM
    run_prrm(query_df, stopwords, dataset_base_path, output_dir, top_n=12, pseudo_top=10)

    # Run Task 5 Evaluation (all results)
    run_task5_evaluation(output_dir=output_dir, relevance_dir='EvaluationBenchmark', k=12)
    
    # Run Task 5 Evaluation (top 10 only)
    run_task5_evaluation_top10(output_dir=output_dir, relevance_dir='EvaluationBenchmark', k=12)