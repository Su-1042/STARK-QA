# STARK-QA Evaluation Report

## System Comparison Overview

| System    |   Total Questions |   Valid Predictions |   Success Rate |   ROUGE-1 |   ROUGE-2 |   ROUGE-L |      BLEU |   BERTScore F1 |   Avg Time (s) |   Total Time (s) |
|:----------|------------------:|--------------------:|---------------:|----------:|----------:|----------:|----------:|---------------:|---------------:|-----------------:|
| STARK-QA  |               100 |                 100 |              1 | 0.464365  | 0.264977  | 0.461884  | 0.0896906 |       0.896839 |     1.70284    |       170.284    |
| Basic-RAG |               100 |                 100 |              1 | 0.155758  | 0.0967507 | 0.147672  | 0.0623067 |       0.838559 |     0.00299988 |         0.299988 |
| Basic-KAG |               100 |                 100 |              1 | 0.0944639 | 0.0161787 | 0.0831411 | 0.0107562 |       0.829843 |     0.00932072 |         0.932072 |

## Detailed Analysis

**Best Overall System:** STARK-QA

### STARK-QA

- **Total Questions Processed:** 100
- **Valid Predictions:** 100
- **Success Rate:** 1.000
- **Average Response Time:** 1.703 seconds
- **ROUGE-1 Score:** 0.464
- **ROUGE-L Score:** 0.462
- **BLEU Score:** 0.090
- **BERTScore F1:** 0.897

**Performance by Question Type:**

- setting: 1.000 (11/11)
- character: 1.000 (9/9)
- action: 1.000 (37/37)
- outcome resolution: 1.000 (9/9)
- feeling: 1.000 (8/8)
- causal relationship: 1.000 (19/19)
- prediction: 1.000 (7/7)

### Basic-RAG

- **Total Questions Processed:** 100
- **Valid Predictions:** 100
- **Success Rate:** 1.000
- **Average Response Time:** 0.003 seconds
- **ROUGE-1 Score:** 0.156
- **ROUGE-L Score:** 0.148
- **BLEU Score:** 0.062
- **BERTScore F1:** 0.839

**Performance by Question Type:**

- setting: 1.000 (11/11)
- character: 1.000 (9/9)
- action: 1.000 (37/37)
- outcome resolution: 1.000 (9/9)
- feeling: 1.000 (8/8)
- causal relationship: 1.000 (19/19)
- prediction: 1.000 (7/7)

### Basic-KAG

- **Total Questions Processed:** 100
- **Valid Predictions:** 100
- **Success Rate:** 1.000
- **Average Response Time:** 0.009 seconds
- **ROUGE-1 Score:** 0.094
- **ROUGE-L Score:** 0.083
- **BLEU Score:** 0.011
- **BERTScore F1:** 0.830

**Performance by Question Type:**

- setting: 1.000 (11/11)
- character: 1.000 (9/9)
- action: 1.000 (37/37)
- outcome resolution: 1.000 (9/9)
- feeling: 1.000 (8/8)
- causal relationship: 1.000 (19/19)
- prediction: 1.000 (7/7)

## Conclusions

This evaluation compares STARK-QA (combining RAG and KAG) against basic RAG and KAG baselines.
The metrics include semantic similarity (ROUGE, BLEU, BERTScore), response time, and question type analysis.
