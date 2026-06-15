# Labs 60-62: NLP fundamentals, text cleaning, and sentiment analysis

import re
import string
from collections import Counter

import nltk

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)
nltk.download("vader_lexicon", quiet=True)

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Lab 60: Linguistics / tokenization
print("-" * 42)
print("Lab 60: NLP Linguistics")
print("-" * 42)

article = (
    "Machine learning is reshaping healthcare and finance. "
    "Pakistan's software industry continues to grow in Islamabad and Lahore."
)

sentences = sent_tokenize(article)
print("Sentence split:")
for i, sentence in enumerate(sentences, 1):
    print(f"  {i}. {sentence}")

tokens = word_tokenize(article)
print(f"\nToken count: {len(tokens)}")
print(f"First tokens: {tokens[:12]}")

tagged = pos_tag(tokens)
tag_map = {
    "NNP": "proper noun", "NN": "noun", "VBZ": "verb",
    "JJ": "adjective", "IN": "preposition", "POS": "possessive",
}
print("\nPOS tags (sample):")
for word, tag in tagged[:10]:
    print(f"  {word:<18} {tag:<6} ({tag_map.get(tag, tag)})")

# Lab 61: Text preprocessing pipeline
print("\n" + "-" * 42)
print("Lab 61: Text Processing")
print("-" * 42)

raw_paragraph = """
Deep learning models learn hierarchical features from data.
NLP pipelines usually clean text before feeding it to a model.
Tokenization, stopword removal, and lemmatization are common steps.
"""

print("Original text:")
print(raw_paragraph.strip())

lowered = raw_paragraph.lower()
no_punct = lowered.translate(str.maketrans("", "", string.punctuation))
word_list = word_tokenize(no_punct)

english_stops = set(stopwords.words("english"))
filtered_words = [w for w in word_list if w not in english_stops and w.isalpha()]

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stemmed = [stemmer.stem(w) for w in filtered_words]
lemmatized = [lemmatizer.lemmatize(w) for w in filtered_words]

print(f"\nAfter cleaning: {len(filtered_words)} tokens")
print("Sample tokens:", filtered_words[:10])
print("\nStem vs lemma (first 6):")
for original, stem, lemma in zip(filtered_words[:6], stemmed[:6], lemmatized[:6]):
    print(f"  {original:<16} stem={stem:<12} lemma={lemma}")

freq = Counter(filtered_words)
print("\nTop words:")
for word, count in freq.most_common(8):
    print(f"  {word:<16} {count}")

# Lab 62: Sentiment + text stats
print("\n" + "-" * 42)
print("Lab 62: Text Analysis")
print("-" * 42)

customer_reviews = [
    "Excellent build quality and quick delivery. Very happy.",
    "Product stopped working after two days. Completely disappointed.",
    "Average experience. It works but nothing special.",
    "Outstanding value for money. Would recommend to friends.",
    "Poor packaging and slow support. Would not buy again.",
]

analyzer = SentimentIntensityAnalyzer()
print("VADER sentiment:")
for review in customer_reviews:
    scores = analyzer.polarity_scores(review)
    compound = scores["compound"]
    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"
    print(f"\n  {review[:50]}...")
    print(f"  compound={compound:.3f} -> {label}")

combined = " ".join(customer_reviews)
all_tokens = [w for w in word_tokenize(combined.lower()) if w.isalpha()]
keywords = [w for w in all_tokens if w not in english_stops]

print("\nCorpus stats:")
print(f"  total words: {len(all_tokens)}")
print(f"  unique words: {len(set(all_tokens))}")
print(f"  lexical diversity: {len(set(all_tokens)) / len(all_tokens):.2%}")

print("\nTop keywords:")
for word, count in Counter(keywords).most_common(6):
    print(f"  {word:<14} {'#' * count} ({count})")

bigrams = list(zip(all_tokens, all_tokens[1:]))
print("\nFrequent bigrams:")
for pair, count in Counter(bigrams).most_common(4):
    print(f"  {' '.join(pair):<22} {count}")

print("\nPipeline: raw text -> tokenize -> clean -> stopwords -> stem/lemma -> analyze")
