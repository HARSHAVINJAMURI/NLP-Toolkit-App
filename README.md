# NLP-Toolkit-App

## Overview

The NLP Toolkit is a Streamlit-based web application for performing basic text processing and analysis using Python. It leverages NLTK, SpaCy, and other NLP libraries to provide features such as tokenization, stemming, lemmatization, POS tagging, n-grams, Bag-of-Words, TF-IDF, parsing, Named Entity Recognition (NER), sentiment analysis, and visualization.

## Features

* Tokenization (words and sentences)
* Stopwords removal
* Stemming (Porter, Lancaster, Snowball)
* Lemmatization
* POS tagging
* N-grams generation
* Bag-of-Words representation
* TF-IDF representation
* Dependency parsing
* Named Entity Recognition (NER)
* Sentiment analysis
* Word frequency plots and WordClouds
* Download original and processed tokens

## Installation

1. Clone the repository:

```bash
git clone <repository_url>
cd <repository_folder>
```

2. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Download the SpaCy English model:

```bash
python -m spacy download en_core_web_sm
```

## Usage

1. Run the Streamlit app:

```bash
streamlit run app.py
```

2. In the sidebar, choose your input source:

   * Paste text
   * Upload `.txt` or `.csv` file (text should be in the first column)

3. Select the desired NLP options and parameters:

   * Tokenization
   * Stopwords removal
   * Stemming / Lemmatization
   * POS tagging
   * N-grams
   * Bag-of-Words / TF-IDF
   * Parsing / NER
   * Sentiment analysis

4. Click **Run NLP** to process the text.

5. Visualizations and downloadable token files are available after processing.

## Requirements

See `requirements.txt` for all dependencies.

## NLTK Data

The app automatically downloads required NLTK datasets if not present:

* punkt
* averaged\_perceptron\_tagger
* wordnet
* omw-1.4
* stopwords


## License

MIT License
