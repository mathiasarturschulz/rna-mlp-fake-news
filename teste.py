
from nltk.corpus import stopwords

# import nltk
# nltk.download()

stop = stopwords.words('english')
sentence = "this is a foo bar sentence"

# print [i for i in sentence.split() if i not in stop]


for i in sentence.split():
    if i not in stop:
        print(i)