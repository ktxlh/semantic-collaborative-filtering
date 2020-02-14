from CSimilarity import CourseSimilarity, Paths
import pickle

from sklearn.manifold import TSNE
from gensim.models import word2vec
import matplotlib.pyplot as plt
import numpy as np
import time

import pandas as pd

from math import log, exp

def SingleCSimTest(n1,n2):
    cs = CourseSimilarity()
    p = Paths()
    name_id = pickle.load(open(p.nameid_path,'rb'))

    c1 = name_id[n1]
    c2 = name_id[n2]

    print(c1,c2,cs.similairy(c1,c2))

class PlotHelper:
    def plot(ks,vs,filename):
        tsne = TSNE(n_components=2, random_state=0, n_iter=10000, perplexity=2)
        np.set_printoptions(suppress=True)
        T = tsne.fit_transform(vs)
        labels = ks

        plt.figure(figsize=(14, 8))
        plt.scatter(T[:, 0], T[:, 1], c='orange', edgecolors='r')
        for label, x, y in zip(labels, T[:, 0], T[:, 1]):
            plt.annotate(label, xy=(x+1, y+1), xytext=(0, 0), textcoords='offset points', fontsize=23)
        plt.savefig(filename, dpi=120)    
        plt.show()


def PlotW2V(wl):    # len(wordlist) >= 1
    w2v_model = word2vec.Word2Vec.load(Paths().model_path)
    similar_words = {search_term: [item[0] for item in w2v_model.wv.most_similar([search_term], topn=5)]
                  for search_term in wl}

    ks = sum([[k] + v for k, v in similar_words.items()], [])
    vs = w2v_model.wv[ks]

    PlotHelper(ks,vs, 'plot/w2v_{0}.png'.format(int(time.time()*100)%10000))


def PlotC2V(cl):    # course list, len >= 1s
    vec = pickle.load(open("course_id_vec-453777-nltk-stop.pkl",'rb'))
    metadata = pd.read_csv("course_id-453777-nltk-stop.csv")
    metadata['Name'] = [row['Subject']+row['Catalog'] for i, row in metadata.iterrows()]

    ks = list()
    vs = list()
    for c in cl:
        rows = metadata.loc[metadata['Name'] == c]
        for cid, row in rows.iterrows():
            #print(row)
            ks.append(row['Long Title'])
            vs.append(vec[cid])

    print(ks)
    #print(vs)

    PlotHelper.plot(ks,vs, 'plot/c2v_453777_{0}_.png'.format(int(time.time()*100)%10000))


def similarC(t):
    id_index = pickle.load(open(Paths.idindex_path,'rb'))
    id_ttl = pickle.load(open(Paths.idttl_path,'rb'))
    sim = pickle.load(open(Paths.resim_path,'rb'))

    for i, id1 in enumerate(id_index):
        cnt = 0
        for j, id2 in enumerate(id_index):
            if j<=i:
                continue
            if sim[i][j] > t and sim[i][j] < 0.99999:
                cnt+=1
        if cnt > 3:
            print(id_ttl[id1])

class ReDist():
    def __init__(self):
        self._MIN_ARG = 0
        self._MAX_ARG = 1
        self.max = max(self.calculate(self._MAX_ARG), self.calculate(self._MIN_ARG))
        self.min = min(self.calculate(self._MAX_ARG), self.calculate(self._MIN_ARG))
        self.range = self.max - self.min

    def rerange(self, x):
        x = min(self._MAX_ARG,x)
        x = max(self._MIN_ARG,x)
        return x

    def exp(self, x, a=10):
        x = self.rerange(x)
        return exp( a * (x-1) )

    def calculate(self, x):
        return self.rerange(x)

def PlotSimDist(n):

    #sim = pickle.load(open(Paths.sim_path,'rb'))
    sim = pickle.load(open("scaled_csim_matrix-nltk-stop-ordinary.pkl",'rb'))
    
    tdist = list(0 for i in range(n+1))
    
    r = ReDist()
    width = r.range/n     # a/n

    for i, row in enumerate(sim):
        for j in range(i,len(row)):
            x = r.calculate(row[j])
            t = int( (x-r.min)/width )
            if t>=0:
                tdist[ t ] += 1
            else:
                print(r.min, x, t)

    dist = list()
    for d in tdist:
        if d>0:
            dist.append( d )
        else:
            dist.append(0)

    fig, ax = plt.subplots()

    index = list( r.min + width*i for i in range(0,n+1))
    bar_width = r.range /(n+1) * 0.9
    opacity = 0.4

    #print(dist[0],len(sim)**2, float(dist[0])/float(len(sim)**2))
    #沒去0的話大部分都0 ><;
    index = list( index[i] for i in range(0,len(index)))
    dist = list( dist[i] for i in range(0,len(dist)))
    ax.bar(index, dist, bar_width, alpha=opacity, color='b')

    ax.set_xlabel('Consine Similarity')
    ax.set_ylabel('Number of Similarities')
    ax.set_title('Distribution of Similarity')
    ax.legend()

    fig.tight_layout()
    fig.savefig('plot/scaled_csim_matrix-nltk-stop-ordinary.png', dpi=120)
    plt.show()

def RedistSim():
    
    """ Deprecated """
    
    print("RedistSim Starts")
    sim = pickle.load(open(Paths.sim_path,'rb'))
    
    rd = ReDist()
    V = len(sim)

    new_sim = np.zeros( (V,V) )
    for i1 in range(V):
        for i2 in range(V):
            new_sim[i1][i2] = rd.calculate(sim[i1][i2])

    pickle.dump(new_sim, open(Paths().resim_path,'wb'), pickle.HIGHEST_PROTOCOL)
    print("RedistSim Ends")

if __name__ == '__main__':
    #PlotW2V(['language','english','programming','code','computer'])
    
    #SingleCSimTest('ECON2103', 'MATH2023')
    
    #p=Paths()
    PlotC2V([
        #c for c in pickle.load(open(p.nameid_path,'rb'))
        #'MATH1012','MATH1013','MATH1014','MATH1023','MATH1024',
        'COMP2711','COMP2711H','COMP3711', 'COMP3711H',
        'LANG1002A','LABU2020','LANG1120','LANG1320'

        #'COMP4411','COMP4421','COMP4431','COMP4441','COMP4451', # Graphic / Multimedia Area
        #'COMP3211','COMP3721','COMP4211','COMP4221','COMP4331'  # Artificial Intelligence / Theory Area
    ])

    
    #similarC(0.985)
    #RedistSim()
    #PlotSimDist(30)
