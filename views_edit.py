#quote koshihara's code#
from flask import Flask, render_template, request
app = Flask(__name__)

import urllib, base64
import io
import re
import collections
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
nltk.download('stopwords')
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import textstat
from textstat.textstat import textstatistics,legacy_round


stopwords.words('english')
en_stops = set(stopwords.words('english'))

nltk.download('punkt')

#def create_n_gram(text):
#    for i in range(1, 6):
#        output = list(ngrams(preprocess(text), i))
#    return output

def preprocess(text):
    strlow = text.lower()
    splstr = strlow.split()
    sedstr = [word.strip('.,!;()[]\n') for word in splstr]
    sedstr = [word.replace("'s", '') for word in sedstr]
    filtered_sent = [w for w in sedstr if not w.lower() in en_stops]
    filtered_sent = []
    for w in sedstr:
        if w not in en_stops:
            filtered_sent.append(w)
    return filtered_sent

def create_n_gram(Q,K):
    total_check_count = 0
    tcc = []
    equal_count = 0
    ec = []
    similality = 0
    sim = []
    out1 = []
    out2 = []
    ngr = []
    for i in range(1, 6):
        ngr.append(i)
        output1 = list(ngrams(preprocess(Q), i))
        output2 = list(ngrams(preprocess(K), i))
        out1.append(output1)
        out2.append(output2)
        #print("This is " + str(i) + "-grams of Q text")
        #print(output1)
        #print("This is " + str(i) + "-grams of K text")
        #print(output2)
        for op1_word in output1:
            total_check_count = total_check_count+1
            tcc.append(total_check_count)
            equal_flag = 0
            for op2_word in output2:
                if op1_word == op2_word:
                    equal_flag = 1
            equal_count = equal_count+equal_flag
            ec.append(equal_count)
            similality = equal_count/total_check_count
            sim.append(similality)
        print('一致した単語数　　：',equal_count)
        print('チェックした単語数：',total_check_count)
        print('一致率(類似度)     　：',equal_count/total_check_count)
        return ec, tcc, sim, out1, out2, ngr
    #ngram_list = []
    #for i in range(1, 6):
    #   output = list(ngrams(preprocess(text), i))
    #   ngram_list.append(output)
    #return ngram_list

#pp stands for preprocess
def pp(sentence):
     strlow = sentence.lower()
     splstr = strlow.split()
     sedstr = [word.strip('@#-.,!;()[]\n') for word in splstr]
     sedstr = [word.replace("'s", '') for word in sedstr]
     return sedstr

def make_list(sentence):
    array = []
    tokens = nltk.pos_tag(pp(sentence))
    out = [lis[1] for lis in tokens]
    #respect https://www.nltk.org/book/ch04.html
    fd = nltk.FreqDist(out)
    #print(fd.most_common(10))
    #this is prnt freaquence postag info. I change to upper 10 used words.
    most_common_words = [word for (word, count) in fd.most_common(10)] # noramally executable
    pos_freq = [i[1] for i in fd.most_common(10)]
    for word in most_common_words:
        array.append(fd.freq(word))
    percentage = array
    #labels = list(most_common_words)
    return most_common_words,percentage,fd,pos_freq

def make_ranking(mcw, fd):
    percentage = 0
    ranks = []
    perce = []
    pos_tags = []
    for rank, word in enumerate(mcw):
        ranks.append(rank+1)
        percentage = fd.freq(word)*100
        perce.append('{0:.2f}'.format(percentage))
        pos_tags.append(word)
        #lists_print.append(print("%3d %6.2f%% %s" % (rank + 1, fd.freq(word) * 100, word)))
    return ranks, perce, pos_tags

def word_count(string):
    return(len(string.strip().split(" ")))

def word_clean(string):
    for char in '-.,\n':
        string = string.replace(char,' ')
    string = string.lower()
    word_list = string.split()
    return(word_list)

def readability(string):
    def word_count(string):
        # Here we are removing the spaces from start and end,
        # and breaking every word whenever we encounter a space
        # and storing them in a list. The len of the list is the
        # total count of words.
        return(len(string.strip().split(" ")))
    #string.strip().split(" ")の部分、文頭の--とかとれていなくて数が合わないの要修正
    
    # Cleaning text and lower casing all words
    def word_clean(string):
        for char in '-.,\n':
            string=string.replace(char,' ')
        string = string.lower()
    # split returns a list of words delimited by sequences of whitespace (including tabs, newlines, etc, like re's \s)
        word_list = string.split()
        return(word_list)
    # Initializing Dictionary
    d = {}
    
    # Count number of times each word comes up in list of words (in dictionary)
    for word in word_clean(string):
        d[word] = d.get(word, 0) + 1
    
    #This list is word freq )
    #print(d)
    
    word_freq = []
    for key, value in d.items():
        word_freq.append((value, key))
    
    word_freq.sort(reverse=True)

    #print(word_clean(string))
    #print(word_count(string))
    def sent_count(string):
        sent = word_clean(string)
        return len(sent)

    def avg_sent_len(text):
        words = word_count(text)
        sent = sent_count(text)
        avg_sent_len = float(words/sent)
        return avg_sent_len

    def avg_syllables_per_word(text):
        syllable = textstatistics().syllable_count(text)
        words = word_count(text)
        ASPW = float(syllable) / float(words)
        return legacy_round(ASPW, 1)

    def difficult_words(text):
        words = []
        sents = word_clean(text)
        for sent in sents:
            words += [str(token)for token in sent]

        diff_words_set = set()

        for word in words:
            syllable_count = textstatistics().syllable_count(word)
            if word not in stopwords.words('english') and syllable_count >= 2:
                diff_words_set.add(word)
        return len(diff_words_set)

    def poly_syllable_count(text):
        cnt = 0
        words = []
        sents = word_clean(text)
        for sent in sents:
            words += [token for token in sent]

        for word in words:
            syllable_count = textstatistics().syllable_count(word)
            if syllable_count >= 3:
                cnt += 1
        return cnt

    def flesch_reading_ease(text):
        FRE = 206.835 - float(1.015 * avg_sent_len(text)) - float(84.6 * avg_syllables_per_word(text))
        return legacy_round(FRE,2)

    def gunning_fog(text):
        per_diff_words = (difficult_words(text) / word_count(text) * 100) + 5
        grade = 0.4 * (avg_sent_len(text) + per_diff_words)
        return grade

    def smog_index(text):

        if sent_count(text) >= 3:
            poly_syllab = poly_syllable_count(text)
            SMOG = (1.043 * (30*(poly_syllab / sent_count(text)))**0.5) + 3.1291
            return legacy_round(SMOG, 1)
        else:
            return 0

    def dale_chall_readability_score(text):
        words = word_count(text)
        cnt = words - difficult_words(text)
        if words > 0:
            per = float(cnt) / float(words) * 100
        diff_words = 100 - per
        raw_score = (0.1579 * diff_words) + (0.0496 * avg_sent_len(text))
        if diff_words > 5:
             raw_score += 3.6365
        return legacy_round(raw_score, 2)    
      
    sedstring = string.replace(" ","")
    lists1 = []
    lists1 += [flesch_reading_ease(string),gunning_fog(string),smog_index(string),dale_chall_readability_score(string),len(sedstring),str(textstat.lexicon_count(string)),str(textstat.sentence_count(string)),str(textstat.syllable_count(string))]
    return lists1

pie_colors = ["r", "c", "b", "m", "y"] # 製品毎の色指定

def histchart(x,y):
     """ 棒グラフを表示する関数
     引数：
     x -- x軸の値
     y -- y軸の値
     """
     #plt.figure(figsize=(25,10)) #描画領域の指定
     #plt.subplots_adjust(wspace=0.2, hspace=0) #間隔指定
     # グラフの装飾
     plt.title("Each Pos-tag number", fontsize = 26) #タイトル
     #plt.xlabel("Top 10 Pos-tag", fontsize = 18) #x軸
     plt.ylabel("Pos-freaquence", fontsize = 26) #y軸
     plt.grid(True) #目盛り線の表示
     plt.tick_params(labelsize=20) #目盛り線のラベルサイズ

     #グラフの描画
     plt.bar(x,y,tick_label=x,align="center",color="c") #棒グラフの描画
     fig = plt.gcf()
     return fig 

def piechart(x,y,c):
     """ 円グラフを表示する関数
     引数 :　c -- 色指定
     """ 
     #plt.figure(figsize=(25,10)) #描画領域の指定
     plt.subplots_adjust(wspace=0.2, hspace=0) #間隔指定
     #グラフの装飾
     plt.title("Pos-tag freaquence TOP10", fontsize = 20, pad=32)
     plt.rcParams['font.size'] = 20.0

     #グラフ描画
     plt.pie(y, labels=x, counterclock=True,autopct="%1.1f%%",colors=c,normalize=True,radius=1.25) #円グラフの描画
     fig = plt.gcf()
     
     return fig

def plot_chart(x,y,z):

    #plt.subplot(1,2,1) #グラフ描画位置の指定
    #histchart(lists[0], lists[3])
    fig_hist = histchart(x,z)
    #plt.subplot(1,2,2) #グラフ描画位置の指定
    #piechart(lists[0], lists[1], pie_colors)
    #fig_pie = piechart(x,y,pie_colors)
    buf = io.BytesIO()
    fig_hist.savefig(buf, format='png')
    #fig_pie.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri = urllib.parse.quote(string)
    return uri
    #cnt = Counter(out)
    #print(cnt.most_common())

def isPassive(sentence):
    beforms = ['am', 'is', 'are', 'been', 'was', 'were', 'be', 'being']               # all forms of "be"
    aux = ['do', 'did', 'does', 'have', 'has', 'had']                                  # NLTK tags "do" and "have" as verbs, which can be misleading in the following section.
    app = ['alarmed', 'aggravated', 'amazed', 'amused', 'annoyed', 'astonished', 'astounded', 'bewildered', 'bored', 'captivated', 'challenged', 'charmed', 'comforted', 'concerned', 'confused', 'convinced', 'depressed', 'devastated', 'disappointed', 'discouraged', 'disgusted', 'distressed', 'disturbed', 'embarrassed', 'enchanted', 'encouraged', 'energise', 'entertained', 'exasperated', 'excited', 'exhausted', 'fascinated', 'flattered', 'frightened', 'frustrated', 'fulfilled', 'gratified', 'horrified', 'humiliated', 'inspired', 'insulted', 'interested', 'intrigued', 'irritated', 'mystified', 'moved', 'overwhelmed', 'perplexed', 'perturbed', 'pleased', 'puzzled', 'relaxed', 'satisfied', 'shocked', 'sickened', 'soothed', 'surprised', 'tempted', 'terrified', 'threatened', 'thrilled', 'tired', 'touched', 'troubled', 'unnerved', 'unsettled', 'upset', 'worried']
    words = pp(sentence)
    #i would like to do  words.remove(-.,\n?)
    tokens = nltk.pos_tag(words)
    #extract pos tagging information
    #out = [lis[1] for lis in tokens]
    dict = {}
    for element in tokens:
        dict[element] = dict.get(element, 0) + 1
    pos_freq = []
    for key, value in dict.items():
        pos_freq.append((key, value))
    #print(pos_freq)
    tags = [i[1] for i in tokens]
    if tags.count('VBN') == 0: # no PP, no passive voice.
        return False
    elif tags.count('VBN') == 1 and 'been' in words:  # one PP "been", still no passive voice.
        return False
    else:
        pos = [i for i in range(len(tags)) if tags[i] == 'VBN' and words[i] != 'been' and words[i] not in app]  # gather all the PPs that are not "been" and are not adjective.
        for end in pos:
            chunk = tags[:end]
            start = 0
            #print(range(len(chunk), 0, -1)
            for i in range(len(chunk), 0, -1):
                last = chunk.pop()
                if last == 'NN' or last == 'PRP':
                    start = i                                                             # get the chunk between PP and the previous NN or PRP (which in most cases are subjects)
                    break
            sentchunk = words[start:end] #words chunk 
            tagschunk = tags[start:end]
            #print(tagschunk)
            verbspos = [i for i in range(len(tagschunk)) if tagschunk[i].startswith('V')] # get all the verbs in between
            if verbspos != []:   # if there are no verbs in between, it's not passive
                for i in verbspos:
                   if sentchunk[i].lower() not in beforms and sentchunk[i].lower() not in aux:  # check if they are all forms of "be" or auxiliaries such as "do" or "have".
                       break
                else:
                    return True
    return False

def p_passive(sentence):
        poscnt = 0
        sentcnt = 0
        sentcnt = sentence.count('.')
        print("This document has " + str(sentcnt) + " sentence")
        sents = nltk.sent_tokenize(sentence)
        for sent in sents:
            if isPassive(sent) == True:
                poscnt += 1
        return poscnt

def test_function():
    return "return text from test_function()"

@app.route("/")
def index():
    return render_template("public/index.html")

@app.route("/about")
def about():
    return """
    <h1 style='color: red;'>I'm a red H1 heading!</h1>
    <p>This is a lovely little paragraph</p>
    <code>Flask is <em>awesome</em></code>
    """
#@app.route("/<input>", methods=["GET", "POST"])
#def inputpage(input):
#    return render_template("input.html", input=input) 

@app.route('/input')
def input_text():
    return render_template('input.html', title="input test", name="input_name")
    
@app.route('/ngram', methods=["GET","POST"])
def show_ngram():
    Q = request.form.get("send_Q") #get n-gram sentence
    K = request.form.get("send_K")
    result = create_n_gram(Q,K)
    #for index,m_words,c_words,sim in zip(result[5], result[0], result[1], result[2]):
    #equal_count = result[0]
    #total_check_count = result[1]
    #similality = result[2]
    #ngrm = result[5]
    merge_list = list(set(result[5] + result[0] + result[1] + result[2]))
    ind = ['1gram','2gram','3gram','4gram','5gram']
    col = ['Ngram','Matched_words','checked_words','similality']
    df = pd.DataFrame(merge_list, index=ind, columns=col)
    df_values = df.values.tolist()
    df_columns = df.columns.tolist()
    df_index = df.index.tolist()
    return render_template('table.html', title="ngram", name="show_name", df_values = df_values, df_columns = df_columns, df_index = df_index, output1=result[3], output2=result[4]) #render_template())
    #return 'input_text: %s' % input_text

#this is readability
@app.route('/read', methods=["GET","POST"])
def show_read():
    Q = request.form.get("send_Q")
    K = request.form.get("send_K")
    df1 = pd.DataFrame({'Q': readability(Q),
                        'K': readability(K)},
                       index=['diff', 'Gunning-Fog', 'SMOG', 'DCRS', 'length', 'total', 'sentence', 'syllable'])
    #return render_template('read.html', title="read", name="show_name", )
    return render_template('read.html', title="read", name="show_name", df1=df1.to_html())
#in order to develope http makes sample 
@app.route('/passive', methods=["GET","POST"])
def show_passive():
    Q = request.form.get("send_Q")
    K = request.form.get("send_K")
    lists1 = make_list(Q)
    lists2 = make_list(K)
    top10_Q = make_ranking(lists1[0], lists1[2])[1]
    Q_pos = make_ranking(lists1[0], lists1[2])[2]
    top10_K = make_ranking(lists2[0], lists2[2])[1]
    K_pos = make_ranking(lists2[0], lists2[2])[2]
    df1 = pd.DataFrame({'percentage': top10_Q,
                        'postag': Q_pos},
                       index=range(1,11))
    df2 = pd.DataFrame({'percentage': top10_K,
                        'postag': K_pos},
                       index=range(1,11))
    # グラフを作成する。
    fig_Q = plot_chart(lists1[0], lists1[1], lists1[3])
    fig_K = plot_chart(lists2[0], lists2[1], lists2[3])
    passive_num_Q = p_passive(Q)
    passive_num_K = p_passive(K)
    #output = io.BytesIO()
    #FigureCanvas(fig).print_png(output)
    return render_template('passive.html', title="passive", name="show_name", df1=df1.to_html(), df2=df2.to_html(), img_Q=fig_Q, img_K=fig_K, passive_num_Q=passive_num_Q, passive_num_K=passive_num_K)
    #return render_template('passive.html', title="passive", name="show_name", df1=df1.to_html())
#this is under-construction
#@app.route('/gram_feature', method["GET","POST"])
#def show_gramf():
#    input_text3 = request.form.get("send_c")
#    result = function(input_text3)
#    return render_template('gram_feature.html', title="gram_feat", name="show_name", )

if __name__ == "__main__":
    app.run(debug=True)
