# IFN647 Information Retrieval Assignment 2

This project implements three different information retrieval models:
1. BM25IR (Task 1)
2. LMRM - Language Model with Relevance Model (Task 2)
3. PRRM - Pseudo-Relevance Feedback (Task 3)

## Project Structure
- `Ass2.py`: Main implementation file containing all IR models
- `test.py`: Script to run all models
- `dataset/`: Contains all dataset files
  - `Dataset101/` to `Dataset150/`: Individual dataset folders
  - `EvaluationBenchmark/`: Evaluation data
- `RankingOutputs/`: Directory for storing ranking results
- `Queries-1.txt`: Query file
- `common-english-words.txt`: Stopwords file

## Requirements
- Python 3.10
- Required packages:
  - numpy
  - pandas
  - stemming

## How to Run
1. Make sure all required packages are installed
2. Run the test script:
```bash
python test.py
```

## Output
The script generates ranking files in the `RankingOutputs` directory:
- `BM25_R{dataset_id}Ranking.dat`
- `LMRM_R{dataset_id}Ranking.dat`
- `My_PRRM_R{dataset_id}Ranking.dat`

Each file contains the top 12 ranked documents for each query. 

## TODO List
- [ ] Task 4 Implementation
  - [ ] Implement new retrieval model
  - [ ] Add evaluation metrics
  - [ ] Compare with existing models
- [ ] Code Optimization
  - [ ] Improve code efficiency
  - [ ] Add error handling
  - [ ] Add logging

