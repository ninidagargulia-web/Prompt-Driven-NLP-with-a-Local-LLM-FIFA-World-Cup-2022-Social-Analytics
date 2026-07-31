# Task 1 — FIFA World Cup Prompt-Driven NLP Pipeline

## Project Overview

This project uses prompt engineering with a local Llama 3.2 model through Ollama to analyze 100 FIFA World Cup 2022 tweets.

The pipeline performs three NLP tasks, with two different prompt strategies evaluated for each task:

1. Sentiment Analysis — Zero-shot vs. Few-shot
2. Topic Classification — Zero-shot vs. System-role
3. Entity Extraction — Zero-shot vs. Structured JSON output

The model is not trained or fine-tuned. Instead, different prompt designs are compared to evaluate how prompt engineering affects NLP performance.

---

## Requirements

- Python 3
- Ollama
- Llama 3.2
- pandas
- ollama Python package

---

## Setup

### 1. Install Ollama

Download and install Ollama from:

https://ollama.com/download

### 2. Download the Llama 3.2 Model

Run:

```bash
ollama pull llama3.2
```

Make sure Ollama is running before executing the Python pipeline.

### 3. Install Python Dependencies

```bash
pip install ollama pandas
```

---

## Project Files

The submission contains the following main files:

- `task1_fifa_nlp_pipeline.py` — main NLP pipeline
- `error_analysis.py` — qualitative error-analysis script
- `task1_selected_tweets_final.csv` — selected dataset containing 100 FIFA World Cup tweets
- `Task1_Report.docx` — final project report
- `README.md` — setup and execution instructions

---

## Running the Project

Make sure the following files are located in the same project directory:

```text
task1_fifa_nlp_pipeline.py
error_analysis.py
task1_selected_tweets_final.csv
```

First, run the main NLP pipeline:

```bash
python task1_fifa_nlp_pipeline.py
```

The script processes all 100 tweets using the different prompt configurations and automatically creates an `outputs` directory.

After the main pipeline has completed successfully, run the error-analysis script:

```bash
python error_analysis.py
```

---

## NLP Tasks and Prompt Strategies

### 1. Sentiment Analysis

Each tweet is classified into one of three sentiment categories:

- Positive
- Negative
- Neutral

Two prompt strategies are compared:

- **Zero-shot:** The model receives only the classification instruction.
- **Few-shot:** The model receives labelled examples before classifying the target tweet.

### 2. Topic Classification

Each tweet is classified into one of three categories:

- Match Play
- Referees
- Logistics/Hosting

Two prompt strategies are compared:

- **Zero-shot:** The model receives the category names and classification instruction.
- **System-role:** The model is assigned a sports-operations analyst role and receives definitions of each topic category.

### 3. Entity Extraction

The model extracts:

- Players
- Teams/countries
- Venues

Two prompt strategies are compared:

- **Zero-shot:** Entities are returned using unrestricted text.
- **Structured output:** Entities are returned using a predefined JSON structure.

---

## Model Configuration

The pipeline uses the following configuration:

```text
Backend: Ollama
Model: llama3.2:latest
Temperature: 0.0
```

The temperature is set to 0.0 to encourage consistent and reproducible model responses.

No API key is required because the Llama model runs locally through Ollama.

---

## Outputs

Running `task1_fifa_nlp_pipeline.py` automatically creates an `outputs` folder containing:

### `task1_results.csv`

Contains the results for all 100 tweets, including:

- Original tweet
- Ground-truth sentiment
- Zero-shot sentiment prediction
- Few-shot sentiment prediction
- Correctness indicators
- Topic reference label
- Zero-shot topic prediction
- System-role topic prediction
- Entity extraction outputs
- Structured JSON validity

### `task1_metrics.json`

Contains the calculated evaluation metrics, including:

- Overall sentiment accuracy
- Sentiment accuracy per class
- Overall topic accuracy
- Topic accuracy per class
- Structured entity-output JSON validity rate
- Average extracted players per tweet
- Average extracted teams per tweet

### `task1_business_summary.txt`

Contains a short LLM-generated summary of the main evaluation findings and their practical implications for tournament media monitoring.

### `task1_error_analysis.md`

Generated after running:

```bash
python error_analysis.py
```

This file contains examples of incorrect sentiment predictions and information about structured entity-extraction parsing failures.

---

## Evaluation Method

### Sentiment Analysis

Sentiment predictions are evaluated against the original sentiment labels provided in the FIFA World Cup dataset.

Overall accuracy and class-specific accuracy are calculated for both the zero-shot and few-shot prompts.

### Topic Classification

The original dataset does not contain manually annotated topic labels.

Therefore, topic predictions are evaluated against keyword-based heuristic reference labels. These labels provide an approximate benchmark for comparing the two prompt strategies and should not be interpreted as manually validated ground truth.

### Entity Extraction

The dataset does not contain manually annotated entities for players, teams, and venues.

Therefore, the structured entity-extraction prompt is evaluated primarily through JSON parse validity and output consistency rather than entity extraction accuracy.

These evaluation limitations are discussed in the final project report.

---

## Dataset

The project uses a curated subset of 100 tweets from the publicly available **FIFA World Cup 2022 Tweets** dataset.

Dataset source:

https://www.kaggle.com/datasets/tirendazacademy/fifa-world-cup-2022-tweets

The selected dataset is included as:

```text
task1_selected_tweets_final.csv
```

The dataset is loaded directly from the CSV file using pandas.

---

## Reproducibility

All 100 selected tweets are processed using both prompt versions for each of the three NLP tasks.

The model temperature is set to `0.0` to reduce variation between runs.

The same dataset and evaluation procedures are used across the prompt comparisons.