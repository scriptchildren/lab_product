import nltk
import re
from nltk import word_tokenize
from nltk.probability import FreqDist
nltk.download('averaged_perceptron_tagger')

f = open("./source/passive.txt", "r", encoding = 'utf-8')
file = f.read()
#lines = """I was born in America in 1989.
 
#preprocesser
def preprocess(lines):
	strlow = lines.lower()
	splstr = strlow.split()
	sedstr = [word.strip('.,!;()[]') for word in splstr]
	sedstr = [word.replace("'s", '') for word in sedstr]
	return sedstr

print(preprocess(file))
 #if find is/was or was/were, count each words appearances
 #also count verb + suffix , for example counted, killed, replaced
 #others passive voice, irregular verb is found by list
pos_tags = nltk.pos_tag(preprocess(file))
#print(pos_tags)

#i would like to extract VBD
#respect https://stackoverflow.com/questions/12845288/grep-on-elements-of-a-list
ext = []
for pos in pos_tags:
	if 'VBN' in pos:
		ext.append(pos)
print(ext)

#After extracting, count freaquence undo
fd = FreqDist(len(w) for w in preprocess(file))
print(sorted(fd.most_common()))

#problem, it is not good to input data to freqdist 
#maybe, it is not shaped to list type data 