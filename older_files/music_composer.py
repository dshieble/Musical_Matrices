import numpy as np
import pandas as pd
import cPickle
from tqdm import tqdm

class MusicComposer():
    #Expects decoded midi input

    seed_len = 1000

    def __init__(self, clf):
        self.clf = clf
        self.seed = list(cPickle.load(open("music_outputs/default_seed.pickle", "rb")))

    def compose(self, notes=100):
        composition = []
        for n in tqdm(range(notes)):
            out = self.clf.predict(self.seed[-self.seed_len:])[0]
            self.seed.append(out)
            if not any(self.seed[-5:]):#any([any(a) for a in self.seed[-5:]]):
                self.seed[-1] = np.random.choice(self.seed)
        return self.seed[-notes:]




