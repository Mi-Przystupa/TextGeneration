import gensim
from gensim.utils import tokenize
from gensim.models import Word2Vec
from handleIMDBSentiment import SentimentDataSet

choices = ['word2vec']
class TextPreprocessing:
    
    def __init__(self, representation, data, stopwords=[]):
        rep = representation.lower()
        self.sw= set(stopwords)
        if (rep == 'word2vec'):
            self.model = self.GetWord2Vec(data)
        else:
            raise Exception("Invalid representation, options: {}".format(choices))
        self.model.save(representation + '.model')

    def CollectData(self, data):
        # assumed data is iterable and returns a tuple of size 1
        # inside the tuple is a string
        sentences = [[token for token in tokenize(d[0], lowercase=True) if token not in self.sw] \
                for d in iter(data)]

        return sentences
    def GetWord2Vec(self, data):
        corpus = self.CollectData(data)
        return Word2Vec(corpus, size=300, window=10, min_count=5) 


if __name__ == "__main__":
    # for debugging
    import pandas as pd
    stopwords = ['<br />']
    data = SentimentDataSet(withLabel=True)
    
    dataset = [d  for d in data]
    dataset = {'review': [d[0] for d in dataset], 'label': [d[1] for d in dataset]}
    pd.DataFrame.from_dict(dataset).to_csv('train.csv')
    
    data = SentimentDataSet(withLabel=True, csv_file='train.csv')
    preprocessing = TextPreprocessing('word2vec', data)
