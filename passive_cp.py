#respect https://github.com/flycrane01/nltk-passive-voice-detector-for-English/blob/master/Passive-voice.py
import nltk
import collections
from nltk import word_tokenize

def isPassive(sentence):
    beforms = ['am', 'is', 'are', 'been', 'was', 'were', 'be', 'being']               # all forms of "be"
    aux = ['do', 'did', 'does', 'have', 'has', 'had']                                  # NLTK tags "do" and "have" as verbs, which can be misleading in the following section.
    app = ['alarmed', 'aggravated', 'amazed', 'amused', 'annoyed', 'astonished', 'astounded', 'bewildered', 'bored', 'captivated', 'challenged', 'charmed', 'comforted', 'concerned', 'confused', 'convinced', 'depressed', 'devastated', 'disappointed', 'discouraged', 'disgusted', 'distressed', 'disturbed', 'embarrassed', 'enchanted', 'encouraged', 'energise', 'entertained', 'exasperated', 'excited', 'exhausted', 'fascinated', 'flattered', 'frightened', 'frustrated', 'fulfilled', 'gratified', 'horrified', 'humiliated', 'inspired', 'insulted', 'interested', 'intrigued', 'irritated', 'mystified', 'moved', 'overwhelmed', 'perplexed', 'perturbed', 'pleased', 'puzzled', 'relaxed', 'satisfied', 'shocked', 'sickened', 'soothed', 'surprised', 'tempted', 'terrified', 'threatened', 'thrilled', 'tired', 'touched', 'troubled', 'unnerved', 'unsettled', 'upset', 'worried']
    words = nltk.word_tokenize(sentence)
    tokens = nltk.pos_tag(words)
    tags = [i[1] for i in tokens]
    if tags.count('VBN') == 0:                                                            # no PP, no passive voice.
        return False
    elif tags.count('VBN') == 1 and 'been' in words:                                    # one PP "been", still no passive voice.
        return False
    else:
        pos = [i for i in range(len(tags)) if tags[i] == 'VBN' and words[i] != 'been']  # gather all the PPs that are not "been".
        #print(len(tags)) #is sum of words include '.,?'
        for end in pos:
            chunk = tags[:end]
            start = 0
            for i in range(len(chunk), 0, -1):
                last = chunk.pop()
                if last == 'NN' or last == 'PRP':
                    start = i                                                             # get the chunk between PP and the previous NN or PRP (which in most cases are subjects)
                    break
            sentchunk = words[start:end] #words chunk 
            tagschunk = tags[start:end]
            #print(pos)
            #print(tagschunk) #tags chunk
            #adjectivepos = [i for i in sentchunk[i].lower() if in app] # get adjectives in between
            #print(adjectivepos)
            verbspos = [i for i in range(len(tagschunk)) if tagschunk[i].startswith('V')] # get all the verbs in between
            count = 0 
            if verbspos != []:   # if there are no verbs in between, it's not passive
                for i in verbspos:
                    if sentchunk[i].lower() not in beforms and sentchunk[i].lower() not in aux:  # check if they are all forms of "be" or auxiliaries such as "do" or "have".
                        break
            #elif adjectivepos != []:
            #    for i in adjectivepos:
            #        if sentchunk[i].lower() not in app:
            #            break
                    if sentchunk[i].lower() in app:
                        break
                else:
                    #count = count + 1
                    return True 
    return False


if __name__ == '__main__':

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
    # "awesome" is wrongly tagged as PP. So the sentence gets a "True".
    sents = nltk.sent_tokenize(samples)
    for sent in sents:
        print(sent + '--> %s' % isPassive(sent))

#plan to add function count sentence have passive voice and caluculate freaquence each pos tags 