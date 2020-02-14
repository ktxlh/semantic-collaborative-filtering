''' 

    To know whether the grades ~ Normal. In order to standardize

    Conclusion: It depends.
        While the ones with more enrollment are more close to Normal,
        the others (e.g. enrollment<20) are actually not.
    => Either set a threshold to standardize, or don't standardize at all.
 
'''
from CSimilarity import CourseSimilarity, Paths

from sklearn.manifold import TSNE
from gensim.models import word2vec
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd

import pickle

from math import log, exp

gdata = pd.read_csv(Paths().grade_path, dialect="excel")

class l2f:
    def __init__(self):
        self.letter2float = {
            'W':-1.0, 'F':0.0, 'D':1.0,
            'C-':1.7, 'C':2.0, 'C+':2.3,
            'B-':2.7, 'B':3.0, 'B+':3.3,
            'A-':3.7, 'A':4.0, 'A+':4.3 }
    def convert(self,l):
        if l in self.letter2float:
            return self.letter2float[l]
        else:
            return -1.0

def GDist(namel, terml):
    if len(namel)!=len(terml):
        return

    N = len(namel)
    M = len(letter2float)

    index2letter = ['F','D','C-','C','C+','B-','B','B+','A-','A','A+']
    letter2index = dict(zip(
        ( item for item in index2letter ),  ( i for i in range(M) )))
    

    nt_id = dict(zip((namel[i]+'_'+str(terml[i]) for i in range(N)), (i for i in range(N))))
    dist = np.zeros((N,M))
    s = np.zeros( N )

    print("GDist: Start iteration")
    for i, row in gdata.iterrows():
        if i%10000==0:
            print("{} rows interated".format(i))

        name = row['Subject'] + row['Catalog']
        term = row['Enroll Term']
        nt = name+'_'+str(term)
        if nt in nt_id:
            grade = row['Grade']
            if grade in letter2index:
                dist[ nt_id[nt] ][ letter2index[grade] ] += 1
                s[ nt_id[nt] ]+=1
            else:
                print("Exception Grade: "+grade)
    print("GDist: Finish iteration")

    for i in range(N): 
        fig, ax = plt.subplots()

        index = index2letter
        bar_width = 0.3
        opacity = 0.4

        ax.bar(index, dist[i], bar_width, alpha=opacity, color='b')

        ax.set_xlabel('Grade')
        ax.set_ylabel('Number of Students')
        ax.set_title('Distribution of Grade: {} {}'.format(namel[i],terml[i]))
        ax.legend()

        fig.tight_layout()

        fig.savefig('plot/gdist_{0}_{1}_s{2}.png'.format(namel[i],terml[i],s[i]), dpi=120)
        #plt.show()


def DataDist():
    """
    def Iterate():
        ts= list()
        for i in range(10,17):
            for j in range(10,50,10):
                ts.append(str(100*i+j))

        t_id = dict(zip(ts, (i for i in range(len(ts)))))
        dist = np.zeros(len(ts))
        
        print("DataDist: Start iteration")
        for i, row in gdata.iterrows():
            if i%10000==0:
                print("{} rows interated".format(i))
            t = str(row['Enroll Term'])
            dist[t_id[t]] += 1
        print("DataDist: Finish iteration")

        pickle.dump(t_id, open('tmp_dd_t_id.pkl','wb'), pickle.HIGHEST_PROTOCOL)
        pickle.dump(ts, open('tmp_dd_ts.pkl','wb'), pickle.HIGHEST_PROTOCOL)
        pickle.dump(dist, open('tmp_dd_dist.pkl','wb'), pickle.HIGHEST_PROTOCOL)
    """
    t_id = pickle.load(open('tmp_dd_t_id.pkl','rb'))
    ts = pickle.load(open('tmp_dd_ts.pkl','rb'))
    dist = pickle.load(open('tmp_dd_dist.pkl','rb'))
    #tdist = list( sum(dist[j] for j in range(i+1)) for i,ti in enumerate(ts) )
    tdist = dist

    fig, ax = plt.subplots()
    bar_width = 1.0
    opacity = 0.4
    ax.bar(ts, tdist, bar_width, alpha=opacity, color='b')
    ax.set_xlabel('Term')
    ax.set_ylabel('Number of Entries')
    ax.set_title('Distribution of Data')
    ax.legend()
    fig.tight_layout()
    fig.savefig('plot/datadist.png', dpi=120)
    plt.show()

if __name__=='__main__':
    #GDist(  ['ECON2123', 'ACCT2200', 'MATH2033', 'SOSC1440'],
    #        [ 1630,       1510,       1330,       1330])
    
    DataDist()