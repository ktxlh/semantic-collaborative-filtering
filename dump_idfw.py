import pickle
from CSimilarity import Paths
from math import log, sqrt, exp
import matplotlib.pyplot as plt

def idf_weight(y,a=1/18.0, b=-6):
    return 1.0 / ( 1.0 + exp( -(a*y+b) ) )

def norm_weight(x):
    return exp(-pow((x-100.0),2)/100000)


def saveplot(l, fname):
    ix = list(i for i in range(len(l)))
    iy = list(
        item[0]
         for i, item in enumerate(l))

    plt.figure(figsize=(14, 8))
    plt.plot(ix, iy, 'b',linewidth=1.0)
    plt.savefig(fname, dpi=120)
    #plt.show()

def dump_idf_weights(word_idf):
    #word_idfweight = dict()
    #for i,word in enumerate(word_idf):
    #    word_idfweight[word] = word_idf[word]
    
    pickle.dump(word_idf, open(p.wordidfw_path,'wb'), pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    p = Paths()
    word_tf = pickle.load(open(p.wordtf_path,'rb'))
    word_idf = pickle.load(open(p.wordidf_path,'rb'))
    #idf準備當CSimilarity裡的weight. 不過需要某個function使其變成0~1 :3
    #順便train tf是要看frequency分佈，看看要不要改stop words

    #tfl = list((word_tf[k],k) for k in word_tf)
    idfl = list((word_idf[k],k) for k in word_idf)
    
    #tfl.sort(reverse=True)
    idfl.sort()
    
    #saveplot(idfl, 'plot/idf_.png')
    dump_idf_weights(word_idf)

