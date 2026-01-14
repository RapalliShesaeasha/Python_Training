import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag, ne_chunk

# Download resources (runs once, safe to keep)
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

def extract_entities(text):
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    tree = ne_chunk(pos_tags)
    return tree

sentence = "Sheshu joined the company in Bangalore"
print(extract_entities(sentence))
