from Ass2 import load_stopwords, load_queries, run_bm25ir, run_lmrm, run_prrm

queries_file = 'Queries-1.txt'         
dataset_base_path = 'dataset'
output_dir = 'RankingOutputs'
stopwords_file = 'common-english-words.txt'

stopwords = load_stopwords(stopwords_file)

query_df = load_queries(queries_file)

run_bm25ir(query_df, stopwords, dataset_base_path, output_dir)

run_lmrm(query_df, stopwords, dataset_base_path, output_dir)

run_prrm(query_df, stopwords, dataset_base_path, output_dir)