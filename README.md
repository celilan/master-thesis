# Master Thesis — Retrieval-Augmented Generation for Research Summary Evaluation

This repository contains the full pipeline and experiments for my master's thesis at OsloMet (Spring 2025), focused on evaluating and improving research summaries using Retrieval-Augmented Generation (RAG) and hallucination detection techniques.

## Project Overview

This project explores how large language models (LLMs) can generate interpretable research summaries while minimizing hallucinations. It includes:
- PDF + metadata extraction
- Data cleaning and preprocessing
- Exploratory data analysis (EDA)
- Retrieval-based summarization pipeline
- Summary evaluation using cosine similarity and iterative refinement with LLM feedback
- Hallucination detection via thresholding and sentence-level analysis

## Folder Structure

```text
master-thesis/
├── data/                     # Raw data (e.g., PDFs)
├── notebooks/                # Jupyter notebooks for EDA, cleaning, extraction
│   ├── data_cleaning.ipynb
│   ├── eda.ipynb
│   ├── pdf_extraction.ipynb
│   └── system_architecture_1.ipynb
├── src/
│   └── retrieval.py          # Script for document retrieval
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

## External Dependencies

This project uses the HaluEval dataset for benchmarking hallucination detection. 

If you are running the code outside of the notebook, clone it manually:

```bash
git clone https://github.com/RUCAIBox/HaluEval.git
```

This command is also included in a code cell in notebooks/system_architecture_1.ipynb, so it's handled automatically when running the notebook.

## How to Run

1. Clone the repository and navigate to the folder:

   ```bash
   git clone https://github.com/celilan/master-thesis.git
   cd master-thesis

2. Create and activate a virtual environment:

    ```bash
    python3 -m venv venv
    source venv/bin/activate

3. Install dependencies

    ```bash
    pip install -r requirements.txt

4. Run the pipeline step-by-step:

    - Start with notebooks/pdf_extraction.ipynb to extract data from PDFs and BibTeX files.
    - Run notebooks/data_cleaning.ipynb to clean the dataset and prepare it for modeling.
    - Use notebooks/eda.ipynb to explore word frequencies, co-authorship, and document lengths.
    - Run notebooks/system_architecture_1.ipynb for the full retrieval, summary generation, and hallucination detection pipeline.

## Accessing Language Models

This project uses pretrained LLMs and embedding models from the Hugging Face Hub, including:

- **Llama3**: `meta-llama/Llama-3.2-1B-Instruct`
- **Falcon3**: `tiiuae/Falcon3-1B-Instruct`
- **Granite**: `ibm-granite/granite-3.0-1b-a400m-instruct`

To access these models, authenticate using your Hugging Face token:

```python
from huggingface_hub import login
login()  # Prompts for your Hugging Face access token
```

You can also download the models locally using the [Hugging Face website](https://huggingface.co/models) or the Hugging Face CLI. Once downloaded, you can load them offline by providing the local path:

```python
tokenizer = AutoTokenizer.from_pretrained("/path/to/local/model")
model = AutoModelForCausalLM.from_pretrained("/path/to/local/model")
```




