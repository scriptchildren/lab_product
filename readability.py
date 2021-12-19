def readability(string):
    #respect https://codeburst.io/python-basics-11-word-count-filter-out-punctuation-dictionary-manipulation-and-sorting-lists-3f6c55420855
    
    Separate="""
    ************************************************************************************************
     """
    
    #string=
    #bands which have connected them with another, and to assume among the powers of the earth, the separate and equal station to which the Laws of Nature and of Nature's God entitle them, a decent respect to the opinions of mankind requires that they should declare the causes which impel them to the separation.  We hold these truths to be
    #self-evident, that all men are created equal, that they are endowed by their Creator with certain unalienable Rights, that among these are Life, Liberty and the pursuit of Happiness.--That to secure these rights, Governments are instituted among Men, deriving their just powers from the consent of the governed, --That whenever any Form of Government becomes destructive of these ends, it is the Right of the People to alter or to abolish it, and to institute new Government, laying its foundation on such principles and organizing its powers in such form, as to them shall seem most likely to effect their Safety and Happiness. """
    
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
    
    print(word_clean(string))
    
    #print(word_list)
    print(Separate)
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
    
    #def ret_result():
    #this_is_sorted_word-freq_list
    print(word_freq)
    print(Separate)
    #this is string's word length 
    print([len(x) for x in string.split()])
    print(Separate)
    #this is length of given string
    sedstring = string.replace(" ","")
    print("The length of the string(not includ space) is :", len(sedstring))
    #redability算出には空白はのぞく
    print(Separate)
    print("'{}'".format(string),"has total words:",word_count(string))
    print(Separate)
    print("This sentence's readability is:",str(word_count(string)/len(string)))
    
if __name__ == '__main__':
    input_text = input('input text : ')
    readability(input_text)

#全体を関数化したら処理した後のtextをword_countに渡しているのでなんとかする

    #return(word_clean(string))

#count space has 1 or 2 after endstop(.!?)
