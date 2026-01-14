from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

words = ["running", "studies", "better"]
stems = [stemmer.stem(word) for word in words]

print(stems)
