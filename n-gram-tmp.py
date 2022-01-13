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

f1 = open("../source/test1.txt", "r", encoding = 'utf-8')
f2 = open("../source/test2.txt", "r", encoding = 'utf-8')
file1 = f1.read()
file2 = f2.read()

#preprocess
#https://www.geeksforgeeks.org/removing-stop-words-nltk-python/
def preprocess(file):
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
    return filtered_sent
#sed_str = re.sub(r"\n", " ", file)
#print(sed_str)
print(preprocess(file1))
print(preprocess(file2))
#print(preprocess(file))
#tokens = [token for token in str(filtered_sent).split(" ") if token != ""]

#This parts is truly functionalize. But, idk how to make function case wants to two variables. 
#def ngrams(file):
#respect https://tat-pytone.hatenablog.com/entry/2019/05/12/211230
total_check_count = 0
equal_count = 0
for i in range(1, 6):
    output1 = list(ngrams(preprocess(file1), i))
    output2 = list(ngrams(preprocess(file2), i))
    print("This is " + str(i) + "-grams of Q text")
    print(output1)
    print("This is " + str(i) + "-grams of K text")
    print(output2)
    for op1_word in output1:
        total_check_count = total_check_count+1
        equal_flag = 0
        for op2_word in output2:
            if op1_word == op2_word:
                equal_flag = 1
        equal_count = equal_count+equal_flag
    print('一致した単語数　　：',equal_count)
    print('チェックした単語数：',total_check_count)
    print('一致率(類似度)     　：',equal_count/total_check_count)
#known document and unknown document
#store two results and compare
#choice random text and compare
#ngram(preprocess(file)))