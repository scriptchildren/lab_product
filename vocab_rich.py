import re
import nltk
from lexicalrichness import LexicalRichness
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
#from nltk import Text

separate = """
******************************************************************
"""

text = """ Measure of textual lexical diversity, computed as the mean length of sequential words in
                a text that maintains a minimum threshold TTR score.#

                                Iterates over words until TTR scores falls below a threshold, then increase factor
                                                counter by 1 and start over. McCarthy and Jarvis (2010, pg. 385) recommends a factor
                                                                threshold in the range of [0.660, 0.750].
                                                                                (McCarthy 2005, McCarthy and Jarvis 2010)
"""

#cleaning
#respect https://pythonexamples.org/python-find-unique-words-in-text-file/
text = text.lower()
words = text.split()
words = [word.strip('.,!;()[]') for word in words]
words = [word.replace("'s", '') for word in words]

#remove stop words
#respect https://www.geeksforgeeks.org/removing-stop-words-nltk-python/
#ここにstop_wordsを処理する用にリストの[]と''と,を除いたもの作る
stop_words = set(stopwords.words('english'))
word_tokens = word_tokenize(str(words))
filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
filtered_sentence = []
for w in word_tokens:
    if w not in stop_words:
            filtered_sentence.append(w)

print(word_tokens)
print(filtered_sentence)

def hapax_func(words):
    list_of_words = re.findall('[a-z]+', str(words))
    freqs = {key: 0 for key in list_of_words}
    for word in list_of_words:
        freqs[word] += 1
    for word in freqs:
        if freqs[word] == 1:
            print(word)

print(separate)            
hapax_func(words)

print(separate)
# instantiate new text object (use the tokenizer=blobber argument to use the textblob tokenizer)
lex = LexicalRichness(text)
Words_list = list(set(lex.wordlist))

print("word list(allow duplicate)" + str(lex.wordlist))
print(separate)
print("number of words is: " + str(lex.words))
print(separate)
print("word list(not duplicate)" + str(Words_list))
print(separate)
print("type-token ration is: " + str(lex.ttr))

#i would like to get hapax legomena