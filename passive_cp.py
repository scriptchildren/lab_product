#respect https://github.com/flycrane01/nltk-passive-voice-detector-for-English/blob/master/Passive-voice.py
import nltk
from collections import Counter
from nltk import word_tokenize
from matplotlib import pyplot as plt
import numpy as np
from nltk.corpus import stopwords

#stopwords.words('english')
#en_stops = set(stopwords.words('english'))

#file = open("../source/Active_vs_passive.txt", "r")
file1 = open("../source/biden_speech.txt")
file2 = open("../source/trump_speech.txt")
Q = file1.read()
K = file2.read()
#sentence = file.read()

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
    most_common_words = [word for (word, count) in fd.most_common()] # noramally executable
    pos_freq = [i[1] for i in fd.most_common()]
    for word in most_common_words:
        array.append(fd.freq(word))
    percentage = array
    labels = list(most_common_words)
    return most_common_words,percentage,fd,pos_freq

lists1 = make_list(Q)
lists2 = make_list(K)
#lists_test = make_list(sentence)
#colors = ['yellowgreen', 'lightgreen', 'darkgreen', 'gold', 'red', 'lightsalmon', 'darkred']
#plt.pie(percentage, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True,  startangle=90)
#plt.axis('equal')
#plt.legend

#print this line pos freq ranking
def make_ranking(mcw, fd):
    for rank, word in enumerate(mcw):
        print("%3d %6.2f%% %s" % (rank + 1, fd.freq(word) * 100, word))

make_ranking(lists1[0], lists1[2])
make_ranking(lists2[0], lists2[2])
#piechart(sentence)

# respect https://ai-inter1.com/python-multichart/
def histchart(x, y):
     """ 棒グラフを表示する関数
     引数：
     x -- x軸の値
     y -- y軸の値
     """
     # グラフの装飾
     plt.title("Each Pos-tag number", fontsize = 26) #タイトル
     plt.xlabel("Pos-tag", fontsize = 26) #x軸
     plt.ylabel("Pos number", fontsize = 26) #y軸
     plt.grid(True) #目盛り線の表示
     plt.tick_params(labelsize=13) #目盛り線のラベルサイズ

     #グラフの描画
     plt.bar(x,y,tick_label=x,align="center",color="c") #棒グラフの描画

def piechart(x,y,c):
     """ 円グラフを表示する関数
     引数 :　c -- 色指定
     """

     #グラフの装飾
     plt.title("Pos-tag freaquence", fontsize = 26)
     plt.rcParams['font.size'] = 13.0

     #グラフ描画
     plt.pie(y, labels=x, counterclock=True,autopct="%1.1f%%",colors=c,normalize=True) #円グラフの描画

def plot_chart(x,y,z):
    #データ準備
    pie_colors = ["r", "c", "b", "m", "y"] # 製品毎の色指定

    #グラフの描画
    plt.figure(figsize=(25,10)) #描画領域の指定
    plt.subplots_adjust(wspace=0.2, hspace=0) #間隔指定

    plt.subplot(1,2,1) #グラフ描画位置の指定
    #histchart(lists[0], lists[3])
    histchart(x,z)

    plt.subplot(1,2,2) #グラフ描画位置の指定
    #piechart(lists[0], lists[1], pie_colors)
    piechart(x,y,pie_colors)
    plt.show()
    #cnt = Counter(out)
    #print(cnt.most_common())

plot_chart(lists1[0], lists1[1], lists1[3])
plot_chart(lists2[0], lists2[1], lists2[3])
#plot_chart(lists_test[0], lists_test[1], lists_test[3])

def isPassive(sentence):
    beforms = ['am', 'is', 'are', 'been', 'was', 'were', 'be', 'being']               # all forms of "be"
    aux = ['do', 'did', 'does', 'have', 'has', 'had']                                  # NLTK tags "do" and "have" as verbs, which can be misleading in the following section.
    app = ['alarmed', 'aggravated', 'amazed', 'amused', 'annoyed', 'astonished', 'astounded', 'bewildered', 'bored', 'captivated', 'challenged', 'charmed', 'comforted', 'concerned', 'confused', 'convinced', 'depressed', 'devastated', 'disappointed', 'discouraged', 'disgusted', 'distressed', 'disturbed', 'embarrassed', 'enchanted', 'encouraged', 'energise', 'entertained', 'exasperated', 'excited', 'exhausted', 'fascinated', 'flattered', 'frightened', 'frustrated', 'fulfilled', 'gratified', 'horrified', 'humiliated', 'inspired', 'insulted', 'interested', 'intrigued', 'irritated', 'mystified', 'moved', 'overwhelmed', 'perplexed', 'perturbed', 'pleased', 'puzzled', 'relaxed', 'satisfied', 'shocked', 'sickened', 'soothed', 'surprised', 'tempted', 'terrified', 'threatened', 'thrilled', 'tired', 'touched', 'troubled', 'unnerved', 'unsettled', 'upset', 'worried']
    words = pp(sentence)
    #i would like to do  words.remove(-.,\n?)
    tokens = nltk.pos_tag(words)
    #extract pos tagging information
    out = [lis[1] for lis in tokens]
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

if __name__ == '__main__':
    # "awesome" is wrongly tagged as PP. So the sentence gets a "True".
    #amount_pos(samples)
    def p_passive(sentence):
        cnt = 0
        cnt = sentence.count('.')
        print("This document has " + str(cnt) + " sentence")
        sents = nltk.sent_tokenize(sentence)
        for sent in sents:
            print(sent + '--> %s' % isPassive(sent))
            
    p_passive(Q)
    p_passive(K)
#plan to add function count sentence have passive voice and caluculate freaquence each pos tags 