import gensim
from gensim.utils import tokenize
from gensim.models import Word2Vec
from handleIMDBSentiment import SentimentDataSet
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from gensim.parsing.preprocessing import  remove_stopwords
choices = ['word2vec']
nltk.download('stopwords')

class TextPreprocessing:
    def __init__(self, representation, data, stop_words=[],filter_top_n=0):
        
        self.rep = representation.lower()
        stop_words = nltk.corpus.stopwords.words('english')
        self.sw = set(stop_words)
        self.filter_top_n = filter_top_n

        if (self.rep == 'word2vec'):
            self.model = self.GetWord2Vec(data)
        else:
            raise Exception("Invalid representation, options: {}".format(choices))
        self.model.save(representation + '.model')

    def TokenizeData(self, data):
        # assumed data is iterable and returns a tuple of size 1
        # inside the tuple is a string
        # my_sentences = [s[0] for s in data if len(s[0]) < 60]
        sentences = [[token for token in tokenize(remove_stopwords(d[0]), lowercase=True) if token not in self.sw] \
                for d in data]

        return sentences

    def WordCounts(self):
        word_counts = {}
        for sentence in self.corpus:
            for s in sentence:
                if s in word_counts:
                    word_counts[s] += 1
                else:
                    word_counts[s] = 1
        return word_counts

    def GetTopN(self, word_counts):
        return null


    def ReplaceTopNWithUNKNOWN(self):
        #r
        if self.filter_top_n > 0:
            word_counts = self.WordCounts()
            top_n = self.GetTopN(word_counts)
            corpus = list(self.corpus)
            for i, c in enumerate(corpus):
                corpus = []

        return





    def GetWord2Vec(self, data):
        self.corpus = self.TokenizeData(data)
        # self.corpus = [c for c in self.corpus if len(c) < 100]
        # self.corpus = self.ReplaceTopNWithUNKNOWN()

        try:
            model = Word2Vec.load('word2vec.model')
        except:
            print('Failed to load')
            model = Word2Vec(self.corpus, size=300, window=10, min_count=10)
        #a 0 vector? 
        model.wv.add('<UNKNOWN>', np.random.rand(300), replace=False)
        model.wv.add('<PADDING>', np.zeros(300), replace=True)
        model.wv.add('<SOS>', np.ones(300), replace=False)
        model.wv.add('<EOS>', np.ones(300) * -1, replace=False)
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

def FilterEntriesByLength(data,indices, savename, length=100):
    sentence_tokens = []
    all_labels = [d[1] for d in data]
    sentence_labels = []
    for i, index_sentence in enumerate(indices):
        if len(index_sentence) <= length:
            sentence_tokens.append(index_sentence)
            sentence_labels.append(all_labels[i])

    dataset = {'review_tokens': sentence_tokens, 'label': sentence_labels}
    np.save(savename, dataset)

def CreateCSV(data, csv_file_name):
    dataset = [d for d in data]
    dataset = {'review': [d[0] for d in dataset], 'label': [d[1] for d in dataset]}
    if '.csv' not in csv_file_name:
        csv_file_name = csv_file_name + '.csv'
    pd.DataFrame.from_dict(dataset).to_csv(csv_file_name)


if __name__ == "__main__":
    # for debugging
    #train_data = SentimentDataSet(withLabel=True)
    #CreateCSV(train_data, 'train.csv')
    #test_data = SentimentDataSet(withLabel=True, path='../aclImdb/test/')
    #CreateCSV(test_data, 'test.csv')

    train_data = SentimentDataSet(withLabel=True, csv_file='train.csv')
    test_data = SentimentDataSet(withLabel=True,csv_file='test.csv')

    stopwords = ['<br />']
    preprocessing = TextPreprocessing('word2vec', train_data, stop_words=stopwords, filter_top_n=True)


    train_indices = preprocessing.ConvertDataToIndices(preprocessing.corpus)

    test_tokens = preprocessing.TokenizeData(test_data)
    test_indices = preprocessing.ConvertDataToIndices(test_tokens)

    FilterEntriesByLength(train_data, train_indices, 'train_data_tokenized')
    FilterEntriesByLength(test_data, test_indices, 'test_data_tokenized')



