import pickle
import numpy as np
import json
from gensim.models import Word2Vec
from gensim.test.utils import common_texts, get_tmpfile
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec


def gloveEmb():
    print('Indexing word vectors.')

    embeddings_index = {}
    f = open('./data/glove.6B.300d.txt', encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))
    pickle.dump(embeddings_index, open('./data/glove.pkl', 'wb'))


def get_Embeddings(num_classes, num_dim, categories_file_name):
    with open('./data/glove.pkl', 'rb') as f:
        glove = pickle.load(f)

    embs = np.zeros((num_classes, num_dim))
    with open(categories_file_name, 'r') as f:
        ann = json.loads(f.read())
        for k, v in ann.items():
            label = k
            index = v
            temp = np.zeros(num_dim)
            label_splits = label.split(" ")

            for l in label_splits:
                temp = temp + glove[l]

            embs[index] = temp / len(label_splits)
    return embs


def testing_pickles(inp_name, adj_file, num_classes):
    t = 0.4
    # glove embedding for each label name. Is the sequence same as category.json?

    with open(inp_name, 'rb') as f:
        inp = pickle.load(f)
        print('inp_name', len(inp), len(inp[0]))

    for i in range(num_classes):
        print(np.sum(inp[i]))

    result = pickle.load(open(adj_file, 'rb'))
    _adj = result['adj']
    _nums = result['nums']
    print('adj_file', len(_adj), len(_adj[0]))

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


# Sequence of these embeddings same as category json

# categories_file_name = "./data/category.json"  # ms-coco dataset (autodownloaded)
# pickle_name = "./data/mscoco_glove_word2vec.pkl"

# categories_file_name = "./data/labels.json"  # ms-coco dataset (autodownloaded)
# pickle_name = "./data/baseline_glove_word2vec.pkl"

categories_file_name = "./data/ms_coco_labels.json"  # ms-coco dataset (autodownloaded)
pickle_name = "./data/visualgnome_glove_word2vec.pkl"

num_classes = 80
num_dims = 300
label_embeddings = get_Embeddings(num_classes, num_dims, categories_file_name)
with open(pickle_name, "wb") as f:
    pickle.dump(label_embeddings, f)

inp_name = 'data/coco/coco_glove_word2vec.pkl'
adj_file = 'data/coco/coco_adj.pkl'
# testing_pickles(inp_name, adj_file, num_classes)
