import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import nltk, io
from collections import Counter
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk import pos_tag
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from textblob import TextBlob

# ----------------------------
# Use local bundled NLTK data
# ----------------------------
nltk.data.path.append("./nltk_data")

# Page layout
st.set_page_config(page_title="NLP Toolkit", layout="wide")
col1, col2 = st.columns([1, 2])
with col1:
    st.title("🧰 NLP Toolkit")
    st.subheader("Basic text processing and analysis using NLTK")
with col2:
    st.markdown("### NLP Tools Step-by-Step")
    st.markdown(
        "• Tokens\n• N-grams\n• Stemming\n• Lemmatization\n• POS tagging\n"
        "• Stopwords removal\n• WordCloud\n• Frequency analysis\n"
        "• Bag-of-Words\n• TF-IDF\n• Sentiment Analysis"
    )

# Sidebar input
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
                txt = f.read().decode("utf-8")

    st.markdown("---")
    st.header("Options")
    show_sentences = st.checkbox("Show sentences", value=True)
    show_paragraphs = st.checkbox("Show paragraphs", value=False)
    lower = st.checkbox("Lowercase / Normalize", value=True)
    rm_stop = st.checkbox("Remove stopwords", value=False)
    stop_lang = st.selectbox("Stopwords language", ["english","none"], index=0)
    stem_choice = st.selectbox("Stemming", ["None","Porter","Lancaster","Snowball"], index=0)
    do_lemm = st.checkbox("Lemmatize", value=False)
    do_pos = st.checkbox("POS tagging", value=False)
    do_ngrams = st.checkbox("N-grams", value=True)
    n_val = st.number_input("n (for n-grams)", min_value=2, max_value=6, value=2)
    do_bow = st.checkbox("Bag-of-Words (BOW)", value=False)
    do_tfidf = st.checkbox("TF-IDF", value=False)
    do_sentiment = st.checkbox("Sentiment Analysis", value=False)
    top_k = st.number_input("Top K frequencies", min_value=5, max_value=200, value=20)
    gen_wc = st.checkbox("Show wordcloud", value=True)
    run = st.button("Run NLP")

# ----------------------------
# Helper functions
# ----------------------------
def split_paragraphs(s): return [p.strip() for p in s.split("\n\n") if p.strip()]

def remove_stopwords(tokens, lang):
    if lang=="none": return tokens
    sw = set(stopwords.words(lang))
    return [t for t in tokens if t.lower() not in sw]

def apply_stem(tokens, choice):
    if choice=="Porter": return [PorterStemmer().stem(t) for t in tokens]
    if choice=="Lancaster": return [LancasterStemmer().stem(t) for t in tokens]
    if choice=="Snowball": return [SnowballStemmer("english").stem(t) for t in tokens]
    return tokens

lemmatizer = WordNetLemmatizer()
def nltk_lemmatize(tokens):
    out=[]
    for token, tag in pos_tag(tokens):
        pos = tag[0].lower()
        wn_tag = wordnet.NOUN
        if pos=="v": wn_tag=wordnet.VERB
        elif pos=="a": wn_tag=wordnet.ADJ
        elif pos=="r": wn_tag=wordnet.ADV
        out.append(lemmatizer.lemmatize(token, wn_tag))
    return out

def freq_table(tokens, k=20):
    c = Counter(tokens)
    return pd.DataFrame(c.most_common(k), columns=["token","freq"])

def make_ngrams(tokens, n): return [" ".join(gram) for gram in ngrams(tokens, n)]

def process_tokens(tokens, lower=False, rm_stop=False, stop_lang="english",
                   stem_choice="None", do_lemm=False):
    proc_tokens = tokens.copy()
    if lower:
        proc_tokens = [t.lower() for t in proc_tokens]
    if rm_stop and stop_lang!="none":
        proc_tokens = remove_stopwords(proc_tokens, stop_lang)
    if stem_choice!="None":
        proc_tokens = apply_stem(proc_tokens, stem_choice)
    if do_lemm:
        proc_tokens = nltk_lemmatize(proc_tokens)
    return proc_tokens

# ----------------------------
# Run NLP
# ----------------------------
if run:
    if not txt.strip():
        st.warning("Please provide some text.")
    else:
        paragraphs = split_paragraphs(txt)
        sents = sent_tokenize(txt)
        words_raw = word_tokenize(txt)

        st.metric("Paragraphs", len(paragraphs))
        st.metric("Sentences", len(sents))
        st.metric("Word tokens (raw)", len(words_raw))

        # Show paragraphs / sentences
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

        # Tokens processing
        tokens_original = words_raw.copy()
        tokens_processed = process_tokens(tokens_original, lower, rm_stop, stop_lang, stem_choice, do_lemm)

        st.subheader("Tokens Comparison")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Original / Without changes**")
            st.write(tokens_original[:200])
        with col2:
            st.markdown("**Processed / With changes applied**")
            st.write(tokens_processed[:200])

        # Frequencies
        tf_original = freq_table(tokens_original, k=top_k)
        tf_processed = freq_table(tokens_processed, k=top_k)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Top tokens (Original)")
            st.dataframe(tf_original, use_container_width=True)
        with col2:
            st.subheader("Top tokens (Processed)")
            st.dataframe(tf_processed, use_container_width=True)

        # N-grams
        if do_ngrams:
            all_ngrams = make_ngrams([t.lower() for t in tokens_processed if t.isalnum()], n_val)
            ng_df = freq_table(all_ngrams, k=top_k)
            st.subheader(f"Top {n_val}-grams (Processed)")
            st.dataframe(ng_df, use_container_width=True)

        # POS tagging
        if do_pos:
            st.subheader("POS Tags (first 200 tokens, Processed)")
            tags = pos_tag(tokens_processed[:200])
            st.dataframe(pd.DataFrame(tags, columns=["token","pos"]), use_container_width=True)

        # Bag-of-Words
        if do_bow:
            vect = CountVectorizer(max_features=top_k)
            X = vect.fit_transform([" ".join(tokens_processed)])
            bow_df = pd.DataFrame(X.toarray(), columns=vect.get_feature_names_out())
            st.subheader("Bag-of-Words (Processed)")
            st.dataframe(bow_df.T, use_container_width=True)

        # TF-IDF
        if do_tfidf:
            tfidf_vect = TfidfVectorizer(max_features=top_k)
            X_tfidf = tfidf_vect.fit_transform([" ".join(tokens_processed)])
            tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=tfidf_vect.get_feature_names_out())
            st.subheader("TF-IDF (Processed)")
            st.dataframe(tfidf_df.T, use_container_width=True)

        # Sentiment Analysis
        if do_sentiment:
            blob = TextBlob(" ".join(tokens_processed))
            st.subheader("Sentiment Analysis (Processed)")
            st.write(f"Polarity: {blob.sentiment.polarity}, Subjectivity: {blob.sentiment.subjectivity}")

        # Visualization
        cols = st.columns([2,1])
        with cols[0]:
            st.subheader("Frequency plot (Processed)")
            plt.figure(figsize=(8,4))
            plt.bar(tf_processed["token"].astype(str), tf_processed["freq"])
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(plt.gcf())
            plt.clf()
        with cols[1]:
            if gen_wc:
                st.subheader("WordCloud (Processed)")
                wc = WordCloud(width=400, height=300, background_color="white").generate(" ".join(tokens_processed))
                plt.imshow(wc, interpolation="bilinear")
                plt.axis("off")
                st.pyplot(plt.gcf())
                plt.clf()

        # Downloads
        tok_buf_proc = "\n".join(tokens_processed)
        st.download_button("Download processed tokens (.txt)", data=tok_buf_proc,
                           file_name="tokens_processed.txt", mime="text/plain")

        tok_buf_orig = "\n".join(tokens_original)
        st.download_button("Download original tokens (.txt)", data=tok_buf_orig,
                           file_name="tokens_original.txt", mime="text/plain")
