import pickle
import numpy as np
import json
from gensim.models import Word2Vec
from gensim.test.utils import common_texts, get_tmpfile
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

def generate_pkls(num_classes):
    # Handle case for multi-word label (take average)

    # model = Word2Vec(common_texts, size=100000000, window=5, min_count=1, workers=4)
    model = KeyedVectors.load_word2vec_format('/home/abajaj/style/PASTEL/data/word2vec/glove.840B.300d.w2v.txt')
    embs = np.zeros((num_classes))
    file_name = "data/category.json"
    with open(file_name, 'r') as f:
        ann = json.loads(f.read())
        for k, v in ann.items():
            label = k
            index = v
            emb = []
            label_splits = label.split(" ")

            for l in label_splits:
                emb.append(model.wv[l])

            embs[index] = np.mean(emb)

    for i in range(num_classes):
        print(np.sum(embs[i]))

t = 0.4
num_classes = 80
generate_pkls(num_classes)
exit(0)


inp_name = 'data/coco/coco_glove_word2vec.pkl'
# glove embedding for each label name. Is the sequence same as category.json?

with open(inp_name, 'rb') as f:
    inp = pickle.load(f)
    print('inp_name', len(inp), len(inp[0]))

adj_file = 'data/coco/coco_adj.pkl'


result = pickle.load(open(adj_file, 'rb'))
_adj = result['adj']
_nums = result['nums']
print('adj_file', len(_adj), len(_adj[0]))
maxf = 0

# Here _adj is diagonal matrix. With all diagonal elements = 0 (co-occurence within same label not considered)

_nums = _nums[:, np.newaxis]
_adj = _adj / _nums
_adj[_adj < t] = 0
_adj[_adj >= t] = 1

# Not diagonal matrix
# _adj is conditional probability matrix (conditional probability, i.e., P(Lj|Li) which denotes the probability
# of occurrence of label Lj when label Li appears)

# Smoothing
_adj = _adj * 0.25 / (_adj.sum(0, keepdims=True) + 1e-6)
_adj = _adj + np.identity(num_classes, np.int)
print('Correlation matrix', len(_adj), len(_adj[0]))
