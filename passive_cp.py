#respect https://github.com/flycrane01/nltk-passive-voice-detector-for-English/blob/master/Passive-voice.py
import nltk
from collections import Counter
from nltk import word_tokenize
#from nltk.corpus import stopwords

#stopwords.words('english')
#en_stops = set(stopwords.words('english'))

samples = '''I like being hunted.
The man is being hunted.
Don't be frightened by what he said.
I assume that you are not informed of the matter.
Please be advised that the park is closing soon.
The book will be released tomorrow.
We're astonished to see the building torn down.
The hunter is literally being chased by the tiger.
He has been awesome since birth.
She has been beautiful since birth.
I am bored.
'''

#file = open("./Active_vs_passive.txt", "r")
#fd = nltk.FreqDist(preprocess(file.read()))
#cumulative = 0.0
#most_common_words = [word for (word, count) in fd.most_common()]
#for rank, word in enumerate(most_common_words):
#    cumulative += fd.freq(word)
#    print("%3d %6.2f%% %s" % (rank + 1, cumulative * 100, word))
#    if cumulative > 0.25:
#        break
#print(most_common_words)

def preprocess(sentence):
     strlow = sentence.lower()
     splstr = strlow.split()
     sedstr = [word.strip('@#-.,!;()[]\n') for word in splstr]
     sedstr = [word.replace("'s", '') for word in sedstr]
     return sedstr

file = open("./Active_vs_passive.txt", "r")
sentence = file.read()
tokens = nltk.pos_tag(preprocess(sentence))
out = [lis[1] for lis in tokens]
#respect https://www.nltk.org/book/ch04.html
fd = nltk.FreqDist(out)
print(fd.most_common(10))
most_common_words = [word for (word, count) in fd.most_common()]
for rank, word in enumerate(most_common_words):
    print("%3d %6.2f%% %s" % (rank + 1, fd.freq(word) * 100, word))

cnt = Counter(out)
print(cnt.most_common())

def isPassive(sentence):
    cntpos = 0
    beforms = ['am', 'is', 'are', 'been', 'was', 'were', 'be', 'being']               # all forms of "be"
    aux = ['do', 'did', 'does', 'have', 'has', 'had']                                  # NLTK tags "do" and "have" as verbs, which can be misleading in the following section.
    app = ['alarmed', 'aggravated', 'amazed', 'amused', 'annoyed', 'astonished', 'astounded', 'bewildered', 'bored', 'captivated', 'challenged', 'charmed', 'comforted', 'concerned', 'confused', 'convinced', 'depressed', 'devastated', 'disappointed', 'discouraged', 'disgusted', 'distressed', 'disturbed', 'embarrassed', 'enchanted', 'encouraged', 'energise', 'entertained', 'exasperated', 'excited', 'exhausted', 'fascinated', 'flattered', 'frightened', 'frustrated', 'fulfilled', 'gratified', 'horrified', 'humiliated', 'inspired', 'insulted', 'interested', 'intrigued', 'irritated', 'mystified', 'moved', 'overwhelmed', 'perplexed', 'perturbed', 'pleased', 'puzzled', 'relaxed', 'satisfied', 'shocked', 'sickened', 'soothed', 'surprised', 'tempted', 'terrified', 'threatened', 'thrilled', 'tired', 'touched', 'troubled', 'unnerved', 'unsettled', 'upset', 'worried']
    words = preprocess(sentence)
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
    print(pos_freq)
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
                    cntpos += 1
                    return True
    return False


if __name__ == '__main__':
    # "awesome" is wrongly tagged as PP. So the sentence gets a "True".
    #amount_pos(samples)
    cnt = 0
    cnt = sentence.count('.')
    print("This document has " + str(cnt) + " sentence")
    sents = nltk.sent_tokenize(sentence)
    for sent in sents:
        print(sent + '--> %s' % isPassive(sent))

#plan to add function count sentence have passive voice and caluculate freaquence each pos tags 