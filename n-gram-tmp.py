#respect
# https://albertauyeung.github.io/2018/06/03/generating-ngram#s.html/
import re
import ast
import nltk
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

nltk.download("punkt")
nltk.download("wordnet")

cont_dict = open("./contractions_dict.txt", "r", encoding = 'utf-8')
cont_dict = cont_dict.read()
cont_dict = ast.literal_eval(cont_dict)
contractions_re = re.compile('(%s)'%'|'.join(cont_dict.keys()))

stopwords.words('english')
en_stops = set(stopwords.words('english'))


f1 = open("../source/test1.txt", "r", encoding = 'utf-8')
f2 = open("../source/test2.txt", "r", encoding = 'utf-8')
#sent = open("./sentence.txt", "r", encoding = 'utf-8')
#cont_dict = open("./contradictions_dict.txt", "r", encoding = 'utf-8')
file1 = f1.read()
file2 = f2.read()
#sent = sent.read()
#cont_dict = cont_dict.read()
#respect https://towardsdatascience.com/text-normalization-for-natural-language-processing-nlp-70a314bfa646
def preprocess(file):
    def expand_contractions(s, cont_dict=cont_dict):
        def replace(match):
            return cont_dict[match.group(0)]
        return contractions_re.sub(replace, s)

    sentence = expand_contractions(file)
    sentence = word_tokenize(sentence)

    def remove_punkt(token):
        return [word for word in token if word.isalpha()]

    sent = remove_punkt(sentence)
    ps = PorterStemmer()
    ps_stem_sent = [ps.stem(words_sent) for words_sent in sent]

    lemmatizer = WordNetLemmatizer()
    lem_sent = [lemmatizer.lemmatize(words_sent) for words_sent in sent]
    filtered_sent = [w for w in lem_sent if not w.lower() in en_stops]
    for w in lem_sent:
        if w not in en_stops:
            filtered_sent.append(w)
    return filtered_sent

#print(sent)
#print(sentence)
#preprocess
#https://www.geeksforgeeks.org/removing-stop-words-nltk-python/
#def preprocess(file):
#    strlow = file.lower()
#    splstr = strlow.split()
#    sedstr = [word.strip('.,!;()[]\n') for word in splstr]
#    sedstr = [word.replace("'s", '') for word in sedstr]
    #word_tokens = word_tokenize(sed_str)
#    filtered_sent = [w for w in sedstr if not w.lower() in en_stops]
#    filtered_sent = []
#    for w in sedstr:
#        if w not in en_stops:
#            filtered_sent.append(w)
#    return filtered_sent
#sed_str = re.sub(r"\n", " ", file)
#print(preprocess(file1))
#print(preprocess(file2))
#print(preprocess(sent))
#print(preprocess(sentence))
#tokens = [token for token in str(filtered_sent).split(" ") if token != ""]

#This parts is truly functionalize. But, idk how to make function case wants to two variables. 
#def ngrams(file):
def make_ngram(Q,K):
    total_check_count = 0
    equal_count = 0
    ec = []
    tcc = []
    sim = []
    #out1 = []
    #out2 = []
    for i in range(1, 6):
        output1 = list(ngrams(preprocess(Q), i))
        output2 = list(ngrams(preprocess(K), i))
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
            similality = equal_count/total_check_count
        print('一致した単語数　　：',equal_count)
        print('チェックした単語数：',total_check_count)
        print('一致率(類似度)     　：',similality)
        ec.append(equal_count)
        tcc.append(total_check_count)
        sim.append(similality)
    #print(ec)
    #print(tcc)
    #print(sim)
    #print(out1)
    #print(out2)
make_ngram(file1,file2)
#known document and unknown document
#store two results and compare
#choice random text and compare
#ngram(preprocess(file)))