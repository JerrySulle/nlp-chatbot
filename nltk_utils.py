import nltk
import numpy as np
#nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    """
    split sentence into array of words/tokens
    a token can be a word or punctuation character, or number
    """
    return nltk.word_tokenize(sentence)


def stem(word):
    """
    stemming = find the root form of the word
    examples:
    words = ["organize", "organizes", "organizing"]
    words = [stem(w) for w in words]
    -> ["organ", "organ", "organ"]
    """
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    #incoming = ["hello","how","are","you"]
    #all_words = ["hi", "hello", "I", "you", "bye","thank","cool"]
    #bag = [ 0, 1, 0, 1, 0, 0, 0]

    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
            #if found set to 1

    return bag

#a = "Can we rent a moped"
#print(a)
#a = tokenize(a)
#print(a)

#words = ["Organize", "organizes", "organizing"]
#stemmed_words = [stem(w) for w in words]
#print(stemmed_words)