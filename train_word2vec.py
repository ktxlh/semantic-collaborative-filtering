"""
Train the word2vec embeddings
"""
import gensim, logging 
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class MySentences(object):
    def __init__(self, fname, N):
        self.fname = fname 
        self.N = N
 
    def __iter__(self):
        with open(self.fname, 'rb') as f:
            for i,line in enumerate(f):      
                if i%N==0:
                    yield gensim.utils.simple_preprocess(line)

if __name__=='__main__':
    
    _inpt = ""
    _size = 80
    _window = 10
    N = 10
    
    input_file = "dataset/wiki_en"+_inpt+".txt"
    sentences = MySentences(input_file, N)
    
    model = Word2Vec(
        sentences,
        size=_size,
        window=_window,
        min_count=5,
        workers=10)
    """ Discussion on parameter 'iter'
        有些人會調iter=? (也就是EPOCH) default是5、大一點會比較準，可是:
        1. Word embeddings準，和用起來好不好常常 沒 有 關係
        2. 調幾次就要跑幾遍(dramatically increase the running time)
        => 有多餘時間再試試別的iter
    """
    model.save("word2vec{0}_s{1}_w{2}.model".format(_inpt,_size,_window))
    