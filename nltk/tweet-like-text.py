import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download("punkt")
nltk.download("wordnet")

text = "Wow!! Loving the new update 😍 #AI #ML"

# Tokenize
tokens = word_tokenize(text.lower())

# Remove non-alphabetic tokens
tokens = [t for t in tokens if t.isalpha()]

# Stemming
stemmer = PorterStemmer()
stemmed = [stemmer.stem(t) for t in tokens]

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized = [lemmatizer.lemmatize(t) for t in tokens]

print("Original Tokens:", tokens)
print("Stemmed:", stemmed)
print("Lemmatized:", lemmatized)
