import nltk
nltk.download("wordnet")

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

words = ["running", "studies", "better"]
lemmas = [lemmatizer.lemmatize(word) for word in words]

print(lemmas)
