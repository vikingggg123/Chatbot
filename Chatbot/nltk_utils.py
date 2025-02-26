import nltk
import nltk.stem.porter
# nltk.download('punkt_tab')
import numpy as np


def Tokenizing(sentence):
    return nltk.word_tokenize(sentence)

def Stemming(word):
    return nltk.stem.porter.PorterStemmer().stem(word=word, to_lowercase=True)

def BagOfWord(tokenizedSentence, allWords):
    tokenizedSentence = [Stemming(w) for w in tokenizedSentence]
    bag = np.zeros(len(allWords),dtype=np.float32)

    for idx, val in enumerate(allWords):
        if val in tokenizedSentence:
            bag[idx] = 1.0

    return bag


tokenize = Tokenizing("hello world i am kien")
word = ['hello','i','wordl','am','kien','nice']
word = []

print(BagOfWord(tokenize, word))
