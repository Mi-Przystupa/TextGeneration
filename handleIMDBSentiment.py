import glob

import pandas as pd
PATH='../aclImdb/train/'


class SentimentDataSet:
    def __init__(self, withLabel=False, path=None, csv_file=None):
        if csv_file is None:
            self.path = PATH if path is None else path
            self.unsup = glob.glob(self.path + 'unsup/*')
            self.pos = glob.glob(self.path +'pos/*')
            self.neg = glob.glob(self.path + 'neg/*')
            self.all = self.assoc(self.unsup, 0) + \
                    self.assoc(self.pos, 1) + \
                    self.assoc(self.neg, 2) 
            self.use_csv = False
        else:
            self.all = pd.read_csv(csv_file)
            self.use_csv = True

        self.withLabel = withLabel

    def SetLabel(self, value):
        self.withLabel = value
    def assoc(self, X, y):
        #X all the data to associate with label y
        return [(x, y) for x in X]
            
    def __iter__(self):
        self.index = 0
        return self

    def path_iter(self):
        curr = self.all[self.index]
        X = curr[0]
        with open(X) as f:
            X = f.read()
        y = curr[1]
        return X, y
    def csv_iter(self):
        X = self.all['review'][self.index]
        y = self.all['label'][self.index]
        return X, y 

    def next(self):
        if self.index < len(self.all):
            if self.use_csv:
                X, y = self.csv_iter()
            else:
                X, y = self.path_iter()

            self.index += 1
            if self.withLabel:
                return [X, y]
            return [X]
        else:
            raise StopIteration
