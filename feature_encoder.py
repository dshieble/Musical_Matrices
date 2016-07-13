import numpy as np
import pandas as pd

class FeatureEncoder():

    n = 128
    def __init__(self):
        self.F = 1
        S = self.stringify(np.zeros(self.n))
        self.arr_to_feature = {S:0}
        self.feature_to_arr = {0:S}
            
    def stringify(self, L):
        return "".join([str(int(i)) for i in L])
    
    def listify(self, S):
        return [int(s) for s in list(S)]
    
    def get_feature(self, arr):
        S = self.stringify(arr)
        if not S in self.arr_to_feature:
            self.arr_to_feature[S] = self.F
            self.feature_to_arr[self.F] = S
            self.F += 1
        out = self.arr_to_feature[S]
        return float(out)

    def get_arr(self, feature):
        S = self.feature_to_arr[int(feature)]
        return self.listify(S)