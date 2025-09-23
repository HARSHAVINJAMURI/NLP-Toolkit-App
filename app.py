import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import nltk, io, os, tempfile
from collections import Counter
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk import pos_tag
from wordcloud import WordCloud

# # ensure required NLTK data
# nltk_data = ["punkt","averaged_perceptron_tagger","wordnet","omw-1.4","stopwords"]
# for d in nltk_data:
#     try:
#         nltk.data.find(d if "/" in d else f"tokenizers/{d}" if d=="punkt" else f"corpora/{d}")
#     except Exception:
#         nltk.download(d)
import nltk
import os

# Use a writable directory for NLTK data (Streamlit Cloud)
nltk_data_dir = "/tmp/nltk_data"
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

# List of required NLTK resources
nltk_resources = [
    "punkt",
    "punkt_tab",  # fix for deployment
    "averaged_perceptron_tagger",
    "wordnet",
    "omw-1.4",
    "stopwords"
]

# Download missing resources to the writable folder
for res in nltk_resources:
    try:
        if res.startswith("punkt"):
            nltk.data.find(f"tokenizers/{res}")
        else:
            nltk.data.find(f"corpora/{res}")
    except LookupError:
        nltk.download(res, download_dir=nltk_data_dir)


st.set_page_config(page_title="NLP Toolkit", layout="wide")
st.title("🧰 NLP Toolkit")
st.title("""tokens · ngrams · stemming · lemmatize · POS · stopwords · wordcloud · frequency""")
st.subheader("Basic text processing and analysis using NLTK")
col1, col2 = st.columns([1, 2])  # left column smaller, right column wider
with col1:
    st.image("https://raw.githubusercontent.com/selva86/datasets/master/nlp.png", width=150)

with col2:
    st.markdown("### NLP Tools Step-by-Step")
    st.markdown("• Tokens")
    st.markdown("• N-grams")
    st.markdown("• Stemming")
    st.markdown("• Lemmatization")
    st.markdown("• POS tagging")
    st.markdown("• Stopwords removal")
    st.markdown("• WordCloud")
    st.markdown("• Frequency analysis")

# input: direct text or file
with st.sidebar:
    st.header("Input")
    src = st.radio("Source", ["Paste text", "Upload file"], index=0)
    if src == "Paste text":
        txt = st.text_area("Paste text here", height=300)
    else:
        f = st.file_uploader("Upload .txt or .csv (text in first column)", type=["txt","csv"])
        txt = ""
        if f:
            if f.type == "text/csv":
                df = pd.read_csv(f, dtype=str, keep_default_na=False)
                if df.shape[1] > 0:
                    txt = "\n\n".join(df.iloc[:,0].astype(str).tolist())
                else:
                    txt = ""
            else:
                txt = f.read().decode("utf-8")

    st.markdown("---")
    st.header("Options")
    show_words = st.checkbox("Show words", value=True)
    show_sentences = st.checkbox("Show sentences", value=True)
    show_paragraphs = st.checkbox("Show paragraphs", value=False)
    do_tokenize = st.checkbox("Tokenize (words)", value=True)
    lower = st.checkbox("Lowercase tokens", value=True)
    rm_stop = st.checkbox("Remove stopwords", value=False)
    stop_lang = st.selectbox("Stopwords language", ["english","none"], index=0)
    stem_choice = st.selectbox("Stemming", ["None","Porter","Lancaster","Snowball"], index=0)
    do_lemm = st.checkbox("Lemmatize", value=False)
    do_pos = st.checkbox("POS tagging", value=False)
    do_ngrams = st.checkbox("N-grams", value=True)
    n_val = st.number_input("n (for n-grams)", min_value=2, max_value=6, value=2)
    top_k = st.number_input("Top K frequencies", min_value=5, max_value=200, value=20)
    gen_wc = st.checkbox("Show wordcloud", value=True)
    run = st.button("Run NLP")

# helper functions
def split_paragraphs(s):
    paras = [p.strip() for p in s.split("\n\n") if p.strip()]
    return paras

def normalize(tok_list):
    if lower:
        return [t.lower() for t in tok_list]
    return tok_list

def remove_stopwords(tokens, lang):
    if lang == "none":
        return tokens
    sw = set(stopwords.words(lang))
    return [t for t in tokens if t.lower() not in sw]

def apply_stem(tokens, choice):
    if choice == "Porter":
        stmr = PorterStemmer()
        return [stmr.stem(t) for t in tokens]
    if choice == "Lancaster":
        stmr = LancasterStemmer()
        return [stmr.stem(t) for t in tokens]
    if choice == "Snowball":
        stmr = SnowballStemmer("english")
        return [stmr.stem(t) for t in tokens]
    return tokens

lemmatizer = WordNetLemmatizer()
def nltk_lemmatize(tokens):
    out=[]
    for token, tag in pos_tag(tokens):
        pos = tag[0].lower()
        wn_tag = wordnet.NOUN
        if pos == "v":
            wn_tag = wordnet.VERB
        elif pos == "a":
            wn_tag = wordnet.ADJ
        elif pos == "r":
            wn_tag = wordnet.ADV
        out.append(lemmatizer.lemmatize(token, wn_tag))
    return out

def freq_table(tokens, k=20):
    c = Counter(tokens)
    most = c.most_common(k)
    df = pd.DataFrame(most, columns=["token","freq"])
    return df

def make_ngrams(tokens, n):
    return [" ".join(gram) for gram in ngrams(tokens, n)]

# run processing
if run:
    if not txt or txt.strip()=="":
        st.warning("Please provide some text (paste or upload).")
    else:
        # paragraphs, sentences, words
        paragraphs = split_paragraphs(txt)
        sents = sent_tokenize(txt)
        words_raw = word_tokenize(txt)

        st.metric("Paragraphs", len(paragraphs))
        st.metric("Sentences", len(sents))
        st.metric("Word tokens (raw)", len(words_raw))

        # optionally show raw lists
        col1, col2 = st.columns(2)
        if show_paragraphs:
            with col1:
                st.subheader("Paragraphs")
                for i,p in enumerate(paragraphs,1):
                    st.markdown(f"**P{i}.** {p[:400]}{'...' if len(p)>400 else ''}")
        if show_sentences:
            with col2:
                st.subheader("Sentences (first 50)")
                for i,s in enumerate(sents[:50],1):
                    st.write(f"{i}. {s}")

        # tokens pipeline
        tokens = words_raw.copy() if do_tokenize else []
        tokens = normalize(tokens)
        if rm_stop and stop_lang!="none":
            tokens = remove_stopwords(tokens, stop_lang)
        if stem_choice!="None":
            tokens = apply_stem(tokens, stem_choice)
        if do_lemm:
            tokens = nltk_lemmatize(tokens)

        # final tokens
        st.subheader("Tokens preview")
        st.write(tokens[:200])

        # frequencies
        tf = freq_table(tokens, k=top_k)
        st.subheader("Top tokens")
        st.dataframe(tf, use_container_width=True)

        # ngrams
        if do_ngrams:
            all_ngrams = make_ngrams([t.lower() for t in word_tokenize(txt) if t.isalnum()], n_val)
            ng_df = freq_table(all_ngrams, k=top_k)
            st.subheader(f"Top {n_val}-grams")
            st.dataframe(ng_df, use_container_width=True)

        # POS tagging
        if do_pos:
            st.subheader("POS Tags (first 200 tokens)")
            tags = pos_tag(word_tokenize(txt))
            st.dataframe(pd.DataFrame(tags[:200], columns=["token","pos"]), use_container_width=True)

        # visualizations
        cols = st.columns([2,1])
        with cols[0]:
            st.subheader("Frequency plot")
            plt.figure(figsize=(8,4))
            plt.bar(tf["token"].astype(str), tf["freq"])
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(plt.gcf())
            plt.clf()
        with cols[1]:
            if gen_wc:
                st.subheader("WordCloud")
                wc = WordCloud(width=400, height=300, background_color="white").generate(" ".join(tokens))
                plt.imshow(wc, interpolation="bilinear")
                plt.axis("off")
                st.pyplot(plt.gcf())
                plt.clf()

        # download results as CSV
        res_buf = io.StringIO()
        tf.to_csv(res_buf, index=False)
        st.download_button("Download token frequencies (CSV)", data=res_buf.getvalue(), file_name="token_freq.csv", mime="text/csv")

        # also allow download of tokens as text
        tok_buf = "\n".join(tokens)
        st.download_button("Download tokens (.txt)", data=tok_buf, file_name="tokens.txt", mime="text/plain")

