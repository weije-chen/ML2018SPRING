import numpy as np
import pandas as pd
from gensim.models import word2vec
import re, sys

data = pd.read_csv(sys.argv[1], sep="\+\+\+\$\+\+\+", engine='python', header=None, names=['label', 'text'])
nolabel_data = pd.read_csv(sys.argv[2], sep="\n", engine='python', header=None, quoting=3, names=['text'])
#
#test_data = pd.read_csv("data/testing_data.txt", sep="\n", skiprows=1, engine='python', names=['text'])
#test_data = test_data['text'].str.split(',', 1 , expand=True)
#test_data['text'] = test_data[1].apply(lambda x: re.sub('[^a-zA-Z0-9\s\?\!]','',x.lower()))

data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-Z0-9\s\?\!]','',x.lower())))
nolabel_data['text'] = nolabel_data['text'].apply((lambda x: re.sub('[^a-zA-Z0-9\s\?\!]','',x.lower())))
test_data['text'] = test_data[1].apply(lambda x: (x.lower()))
data['text'] = data['text'].apply((lambda x: (x.lower())))
nolabel_data['text'] = nolabel_data['text'].apply((lambda x: (x.lower())))

two_text = np.append(data['text'].str.split().values, nolabel_data['text'].str.split().values)
all_text = np.append(test_data['text'].str.split().values,two_text)


word_dim = 120

model = word2vec.Word2Vec(all_text, sg=1,min_count=5, size=word_dim)

model.save('emb')
