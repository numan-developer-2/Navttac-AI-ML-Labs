# Lab 70: Advanced NLP — NER, BoW, TF-IDF, Word2Vec (NAVTTC Week 6, 9)

import re
from collections import Counter

import nltk
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)
nltk.download("maxent_ne_chunker", quiet=True)
nltk.download("words", quiet=True)

from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.chunk import tree2conlltags

try:
    from gensim.models import Word2Vec
    HAS_GENSIM = True
except ImportError:
    HAS_GENSIM = False

print("-" * 42)
print("Lab 70: NER, BoW, TF-IDF, Word2Vec")
print("-" * 42)

text = (
    "Apple Inc. was founded by Steve Jobs in Cupertino. "
    "Microsoft Azure powers AI solutions in Seattle."
)

tokens = word_tokenize(text)
tagged = pos_tag(tokens)
tree = ne_chunk(tagged)

print("Named Entity Recognition (rule-based NLTK):")
for chunk in tree:
    if hasattr(chunk, "label"):
        entity = " ".join(c[0] for c in chunk)
        print(f"  {entity:<20} -> {chunk.label()}")

# BoW and TF-IDF
docs = [
    "machine learning models learn patterns from data",
    "deep learning uses neural networks for complex tasks",
    "natural language processing analyzes human text",
    "azure cloud provides scalable ai services",
]

bow = CountVectorizer(stop_words="english")
bow_matrix = bow.fit_transform(docs)
print("\nBag of Words vocabulary (sample):", bow.get_feature_names_out()[:8])
print("BoW matrix shape:", bow_matrix.shape)

tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(docs)
sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
print(f"\nDocument similarity (doc0 vs doc1): {sim:.3f}")

# Word2Vec
if HAS_GENSIM:
    sentences = [word_tokenize(re.sub(r"[^a-zA-Z\s]", "", d.lower())) for d in docs]
    w2v = Word2Vec(sentences, vector_size=50, window=3, min_count=1, workers=1, seed=17)
    if "learning" in w2v.wv and "neural" in w2v.wv:
        sim_w = w2v.wv.similarity("learning", "neural")
        print(f"Word2Vec similarity (learning, neural): {sim_w:.3f}")
    print(f"Word2Vec vocabulary size: {len(w2v.wv)}")
else:
    print("\nInstall gensim for Word2Vec: pip install gensim")

# Word frequency
words = [w.lower() for w in word_tokenize(" ".join(docs)) if w.isalpha()]
freq = Counter(words).most_common(8)
print("\nTop word frequencies:")
for word, count in freq:
    print(f"  {word:<12} {count}")

print("\nNLP pipeline covered: tokenize -> POS/NER -> BoW/TF-IDF -> embeddings")
