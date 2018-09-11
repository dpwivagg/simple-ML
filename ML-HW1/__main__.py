from part1 import *
import numpy as np


class credit_tree():
    def __init__(self, filename):
        self.filename = filename

    def load_dataset(self):
        '''
            Load the credit risk dataset
        '''
        alldata = np.loadtxt(self.filename, delimiter=',', dtype=str)

        self.attributes = alldata[0, 1:-1]
        self.X = alldata[1:, 1:-1].T
        self.Y = alldata[1:, -1].T

    def create_tree(self):
        self.t = Tree.train(self.X, self.Y)

    def print_tree(self, t=None, layer=0):
        if not t:
            t = self.t

        if t.isleaf:
            print(":", t.p, " risk", end='')

        else:
            for key, value in t.C.items():
                print('\n', '|' * layer, self.attributes[t.i], '=', key, end='')
                self.print_tree(t.C[key], layer+1)

    def inference(self, data):
        return Tree.inference(self.t, data)

    def make_and_show(self):
        self.load_dataset()
        self.create_tree()
        self.print_tree()

######################################################

credit1 = credit_tree('credit.csv')
credit1.make_and_show()

tom_risk = credit1.inference(np.array(['low', 'low', 'no', 'yes', 'male']))
ana_risk = credit1.inference(np.array(['low', 'medium', 'yes', 'yes', 'female']))

print("\n\nTom's credit risk: ", tom_risk, "\nAna's credit risk: ", ana_risk)

credit2 = credit_tree('credit_sofia_high_risk.csv')
credit2.make_and_show()
