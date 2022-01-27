import textstat
from nltk.corpus import stopwords
from textstat.textstat import textstatistics,legacy_round
#respect https://codeburst.io/python-basics-11-word-count-filter-out-punctuation-dictionary-manipulation-and-sorting-lists-3f6c55420855
#https://www.geeksforgeeks.org/readability-index-pythonnlp/
    
Separate="""
************************************************************************************************
"""

string="""
bands which have connected them with another, and to assume among the powers of the earth, the separate and equal station to which the Laws of Nature and of Nature's God entitle them, a decent respect to the opinions of mankind requires that they should declare the causes which impel them to the separation.  We hold these truths to be self-evident, that all men are created equal, that they are endowed by their Creator with certain unalienable Rights, that among these are Life, Liberty and the pursuit of Happiness.--That to secure these rights, Governments are instituted among Men, deriving their just powers from the consent of the governed, --That whenever any Form of Government becomes destructive of these ends, it is the Right of the People to alter or to abolish it, and to institute new Government, laying its foundation on such principles and organizing its powers in such form, as to them shall seem most likely to effect their Safety and Happiness. """

#path = '../source/Active_vs_passive.txt'
#file = open('../source/Active_vs_passive.txt','r')
file1 = open('../source/biden_speech.txt')
file2 = open('../source/trump_speech.txt')
Q = file1.read()
K = file2.read()
#sentence = file.read()

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
    print("The sentence's diffficult is:",flesch_reading_ease(string))
    print("The Gunning-Fog's value is:",gunning_fog(string))
    print("The SMOG-index's value is:",smog_index(string))
    print("The dale_chall_readability_score is:",dale_chall_readability_score(string))
      
    #def ret_result():
    #this_is_sorted_word-freq_list
    #print(word_clean(sentence))
    #print(word_freq)
    #print(Separate)
    #this is string's word length
    #print([len(x) for x in string.split()])
    #print(Separate)
    #this is length of given string
    sedstring = string.replace(" ","")
    print("The length of the string(not includ space) is :", len(sedstring))
    #redability算出には空白はのぞく
    #print(Separate)
    #print("'{}'".format(string),"has total words:",word_count(string))
    print("This sentence has total words:",str(textstat.lexicon_count(string)))
    #print(Separate)
    print("This document has " + str(textstat.sentence_count(string)) + " sentences.")
    #print(Separate)
    print("This sentences has " + str(textstat.syllable_count(string)) + " syllables")

readability(Q)
print(Separate)
readability(K)
#全体を関数化したら処理した後のtextをword_countに渡しているのでなんとかする
    #return(word_clean(string))
#count space has 1 or 2 after endstop(.!?)