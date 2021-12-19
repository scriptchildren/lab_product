#respect 
# https://albertauyeung.github.io/2018/06/03/generating-ngram#s.html/

import re
import nltk
nltk.download('stopwords')
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stopwords.words('english')
en_stops = set(stopwords.words('english'))

f = open("./source/passive.txt", "r", encoding = 'utf-8')
file = f.read()

#preprocess
strlow = file.lower()
splstr = strlow.split()
sedstr = [word.strip('.,!;()[]\n') for word in splstr]
sedstr = [word.replace("'s", '') for word in sedstr]
#word_tokens = word_tokenize(sed_str)
filtered_sent = [w for w in sedstr if not w.lower() in en_stops]
filtered_sent = []
for w in sedstr:
	if w not in en_stops:
		filtered_sent.append(w)
print(filtered_sent)
#sed_str = re.sub(r"\n", " ", file)
#print(sed_str)

#tokens = [token for token in str(filtered_sent).split(" ") if token != ""]
for i in range(1, 5):
    print("This is " + str(i) + "-grams")
    output = list(ngrams(filtered_sent, i))
    print(output)
    #known document and unknown document 
    #store two results and compare
    #choice random text and compare 