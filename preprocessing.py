import gensim
from gensim.utils import tokenize
from gensim.models import Word2Vec
from handleIMDBSentiment import SentimentDataSet
import numpy as np
choices = ['word2vec']


class TextPreprocessing:
    def __init__(self, representation, data, stopwords=[]):
        
        self.rep = representation.lower()
        self.sw = set(stopwords)

        if (self.rep == 'word2vec'):
            self.model = self.GetWord2Vec(data)
        else:
            raise Exception("Invalid representation, options: {}".format(choices))
        self.model.save(representation + '.model')

    def TokenizeData(self, data):
        # assumed data is iterable and returns a tuple of size 1
        # inside the tuple is a string
        sentences = [[token for token in tokenize(d[0], lowercase=True) if token not in self.sw] \
                for d in iter(data)]

        return sentences

    def GetWord2Vec(self, data):
        self.corpus = self.TokenizeData(data)
        try:
            model = Word2Vec.load('word2vec.model')
        except:
            print('Failed to load')
            model = Word2Vec(self.corpus, size=300, window=10, min_count=5)
        #a 0 vector? 
        model.wv.add('<UNKNOWN>', np.random.rand(300), replace=False)
        model.wv.add('<PADDING>', np.zeros(300), replace=True)
        return model


    def ConvertDataToIndices(self, X, y=None):
        if self.rep == 'word2vec':
            model = self.model.wv
            transformer = lambda x: model.vocab.get(x).index \
                    if x in model.vocab else model.vocab.get('<UNKNOWN>').index 
            ret = [None] * len(X)
            for i, x in enumerate(X):

                tokens = [transformer(w) for w in x]
                ret[i] = tokens

            X = ret
            #padded_sequences = self.pad_sequences(X)
            #return padded_sequences
            return X
        else:
            print('invalid representation returning data unchanged')
            return X




if __name__ == "__main__":
    # for debugging
    import pandas as pd
    stopwords = ['<br />']
    data = SentimentDataSet(withLabel=True)

    dataset = [d for d in data]
    dataset = {'review': [d[0] for d in dataset], 'label': [d[1] for d in dataset]}
    pd.DataFrame.from_dict(dataset).to_csv('train.csv')

    data = SentimentDataSet(withLabel=True, csv_file='train.csv')
    preprocessing = TextPreprocessing('word2vec', data)

    indices = preprocessing.ConvertDataToIndices(preprocessing.corpus)
    #print(len(indices))
    print(indices[1])
    dataset = {'review_tokens': [i for i in indices], 'label': [d[1] for d in data]}
    np.save('tokenized_dataaset', dataset)
