""" 

    Calculate the course similairy and provide the similarity API 

"""
#import gensim
import logging
import pandas as pd 
import pickle
import nltk
import string
from math import sqrt
from nltk.corpus import stopwords, webtext
#from local_stopwords import words
#from gensim.models import Word2Vec
import numpy as np
import csv
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.WARN)
#stop_words = set(stopwords.words('english') + list(string.punctuation) + words.course)
#stop_words = set(stopwords.words('english') + list(string.punctuation) + words.manual)
stop_words = set(stopwords.words('english') + list(string.punctuation))

class Paths:
    #root_path = "/Users/shanglinghsu/OneDrive - HKUST Connect/Data Mining/DataMining/dataset/"
    root_path = ""
    grade_path = root_path + "Grade.csv"
    cinfo_path = root_path + "CInfo.csv"

    idvec_path = root_path + "id_vec_4538_s80_w10.pkl"
    idttl_path = root_path + "id_ttl.pkl"
    idname_path = root_path + "id_name.pkl"
    #idindex_path = root_path + "id_index.pkl"
    
    nameid_path = root_path + "name_id.pkl"
    sim_path = root_path + "sim_4538_s80_w10.pkl"
    resim_path = root_path + "resim.pkl"
    
    ''' Both norm and idf-sigmoid weight yields horrible results :( '''
    #wordtf_path = root_path + "word_tf.pkl"
    #wordidf_path = root_path + "word_idf.pkl"
    #wordidfw_path = root_path + "word_idfw.pkl"
    
    
    term2id = root_path + 'term2id.pkl'
    terms = root_path + 'terms.pkl'
    course2id = root_path + 'course2id.pkl'
    courses = root_path + 'courses.pkl'
    student2id = root_path + 'student2id.pkl'
    students = root_path + 'students.pkl'
    
    #grades = root_path + 'grades.pkl'   # [student_id][term_id][course_id] = grade
    flat = root_path + 'flat.pkl'       # [student_id][course_id] = (grade, term)
    sample = root_path + 'sample.pkl'   # flat of random sampled students 
    smean = root_path + 'smean.pkl'   # sample students' mean
    sgsq = root_path + 'sgsq.pkl'     # sqrt(∑(g - µg)^2)
    studsim = root_path + 'studsim.pkl'
    studsim_2 = root_path + 'studsim_2.pkl' #不除以 ∑weight

    li_pred = root_path + 'li_pred.pkl'

    # changed to the larger one!
    model_path = "wiki/word2vec_453777_s80_w10.model"


class CourseSimilarity:

    def __init__(self):
        self._sim = pickle.load(open(Paths().sim_path, 'rb'))
        self._cidindex = pickle.load(open(Paths().course2id, 'rb'))
    
    def similairy(self, cid1, cid2):
        if (cid1 not in self._cidindex) or (cid2 not in self._cidindex):
            logging.warning("cid not in cidindex: "+str(cid1)+" or "+str(cid2))
            return 0.0

        i1 = self._cidindex[cid1]
        i2 = self._cidindex[cid2]
        if i2 < i1:
            i1, i2 = i2, i1
        
        return self._sim[i1][i2]


def tokenize(text, counted = False):

    ''' Helper: Tokenize a sentence '''
    
    wc1 = 0
    wc2 = 0
    tokens = []
    model = Word2Vec.load("word2vec_s80_w10.model")

    for word in nltk.word_tokenize(text):
        word = word.lower()
        wc2 += 1
        if word not in stop_words and not word.isnumeric():
            if word in model.wv:
                tokens.append(word)
                wc1 += 1
            #tokens.append(word)
            #wc1 += 1
    if counted:
        return tokens, wc1, wc2
    else:
        return tokens


def text2Vec(text):
    
    ''' Helper: Embed a piece of text '''
    
    p = Paths()
    vecs = list()
    model = Word2Vec.load("word2vec_s80_w10.model")

    tokenized, wc1, wc2 = tokenize(text, True)
    for word in tokenized:
        wv = model.wv[word]
        vecs.append(wv)

    if len(vecs) == 0:
        return None, wc1, wc2

    return list( np.average( list(vecs[i][j] for i in range(len(vecs))) ) 
                for j in range(len(vecs[0])) ), wc1, wc2



def desc2Vec(title, desc, weight=0.1):
    
    ''' Helper: Embed a course' title AND description '''
    
    ta, c1, c2 = text2Vec(title)
    da, c3, c4 = text2Vec(desc)
    wc1 = c1 + c3
    wc2 = c2 + c4

    if ta == None and da != None:
        return da, wc1, wc2
    elif da == None and ta != None:
        return ta, wc1, wc2
    elif da == None and ta == None:
        #print(title, desc, "returns None")
        return None, wc1, wc2
    else:
        return list( weight*t + (1-weight)*da[i]  for i, t in enumerate(ta)), wc1, wc2



def id2Vec_train():
    
    ''' Embed all distinct course description and return their semantic representation '''
    
    print("id2Vec_train Starts")
    p = Paths()
    id_vec = dict()
    cinfo = pd.read_csv("dataset/CInfo.csv", dialect="excel")

    def match_course_id():
        pass
        # Match the cinfo with id
        """
        for dummy, row in grade.iterrows():
            name = row['Subject'] + row['Catalog']
            cid = row['Course ID']
            if name not in name_id:
                name_id[name] = cid
            if cid not in id_name:
                id_name[cid] = name
        """
    
    name_id = pickle.load(open("dataset/name_id.pkl",'rb'))
    
    df = pd.DataFrame()
    vecs = []
    titles = []
    descriptions = []
    subjects = []
    catalogs = []
    ids = []
    c=0
    wc1 = 0
    wc2 = 0
    for i, row in cinfo.iterrows():
        key = row['Subject'] + row['Catalog']
        if key in name_id:
            cid = name_id[key]
            if cid not in id_vec:
                vec, c1, c2 = desc2Vec(row['Long Title'], row['DESCRIPTION'])
                if vec == None:
                    # 這一步會再換一套id
                    print(key, vec)
                    continue
                wc1 += c1
                wc2 += c2
                id_vec[cid] = vec
                vecs.append(vec)
                ids.append(c)
                titles.append(tokenize(row['Long Title']))
                descriptions.append(tokenize(row['DESCRIPTION']))
                subjects.append(row['Subject'])
                catalogs.append(row['Catalog'])
                c+=1
                print(c,"courses iterated ("+key+")")
    print("%d courses are available."%c)
    
    df['ID'] = ids
    df['Subject'] = subjects
    df['Catalog'] = catalogs
    df['Long Title'] = [ ' '.join(title) for title in titles]
    df['DESCRIPTION'] = [ ' '.join(description) for description in descriptions]
    df.to_csv("course_id-453777-nltk-stop.csv", index=False)
    pickle.dump(vecs, open("course_id_vec-453777-nltk-stop.pkl",'wb'), pickle.HIGHEST_PROTOCOL) 
    print("vecs are of shape:",len(vecs),len(vecs[0]))

    def see_ignored_records():
        pass
        ''' To see which courses are ignored from the past records '''
        '''
        not_in = []
        for _, name in enumerate(name_id):
            if name_id[name] not in id_vec:
                not_in.append(name)
        not_in.sort()
        cnt=0
        with open("past - now.txt",'w') as f:
            for name in not_in:
                f.writelines(name+'\n')
                cnt+=1
        print(cnt)
        '''
    
    print("Words used:", wc1, "of", wc2, "(", wc1/wc2, "%)")

    print("id2Vec_train Ends")
    return id_vec

def cosine_sim(u,v):

    ''' Helper: Calculate the cosine value of two vectors '''

    cosine_score = np.dot(u,v) / (sqrt(np.dot(u,u)) * sqrt(np.dot(v,v)) )
    return cosine_score


def scale_id_vec():
    ''' 
    vec-mean, 0 if <0 
    '''
    
    vecs = pickle.load(open("course_id_vec-453777-nltk-stop.pkl",'rb'))
    print("scale_id_vec(): vecs are of shape",len(vecs),len(vecs[0]))
    df = pd.DataFrame()
    
    N = len(vecs)
    V = len(vecs[0])
    df['vec'] = vecs

    for _, row in df.iterrows():
        if type(row['vec']) != list:
            print(row, type(row['vec']))
            row['vec'] = [0.0 for i in range(V)]
    
    #maxs = [max( row['vec'][i] for _, row in df.iterrows() ) for i in range(V) ]
    #mins = [min( row['vec'][i] for _, row in df.iterrows()) for i in range(V)]
    #means = [ np.mean( [row['vec'][i] for _, row in df.iterrows()]) for i in range(V)]
    
    print("vec-mean, 0 if <0")
    #df['vec'] = [ [(row['vec'][i]-means[i]) for i in range(V)] for _, row in df.iterrows()]
    #new_csim = [[max(cosine_sim(vec1, vec2),0) for vec2 in df['vec']] for vec1 in df['vec']] 
    new_csim = [[cosine_sim(vec1, vec2) for vec2 in df['vec']] for vec1 in df['vec']] 
    
    ndf = pd.DataFrame()
    ciddf = pd.read_csv("course_id-453777-nltk-stop.csv")
    ndf['SubjectCatalog'] = [row['Subject']+row['Catalog'] for i, row in ciddf.iterrows()]
    ndf.set_index('SubjectCatalog')
    for i, row in ciddf.iterrows():
        ndf[row['Subject']+row['Catalog']] = new_csim[i]

    ndf.to_csv("scaled_csim_matrix-nltk-stop-ordinary.csv", index=False)
    pickle.dump(new_csim, open("scaled_csim_matrix-nltk-stop-ordinary.pkl",'wb'), pickle.HIGHEST_PROTOCOL) 

# 其實是union
def intersection_course():
    
    print("intersection_course Starts")
    
    cinfo = pd.read_csv(Paths().cinfo_path)
    past_courses = set()
    for i, row in cinfo.iterrows():
        past_courses.add(row['Subject'] + row['Catalog'])
    df = pd.read_csv("course_id.csv")
    vecs = pickle.load(open("course_id_vec.pkl",'rb'))

    # 這邊又換一套id了 這樣list才O(1)
    course_union = []
    course_union_2id = dict()
    new_vecs = []
    for old_id, row1 in df.iterrows():
        key1 = row1['Subject'] + row1['Catalog']
        if key1 not in past_courses:
            continue
        if type(vecs[old_id]) != type(vecs[0]):
            print(key1, vecs[old_id])
            continue
        new_id = len(course_union)
        course_union_2id[key1] = new_id
        course_union.append(key1)
        new_vecs.append(vecs[old_id])
    
    pickle.dump(course_union_2id, open( "course_union_2id.pkl",'wb'), pickle.HIGHEST_PROTOCOL)
    pickle.dump(new_vecs, open( "new_course_vecs.pkl",'wb'), pickle.HIGHEST_PROTOCOL)
    print("intersection_course Ends")

def CSim_pkl2csv():
    sim = pickle.load(open("scaled_csim_matrix.pkl",'rb'))
    with open("scaled_csim_matrix.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(sim) 


def similarity_train():

    ''' Train the similarity between course vectors
            Out: (dense) matrix[cid1][cid2] = similarity
    '''

    print("similarity_train Starts")
    
    course_union_2id = pickle.load(open("course_union_2id.pkl",'rb'))
    new_vecs = pickle.load(open("new_course_vecs.pkl",'rb'))
    sim = np.zeros( (len(new_vecs), len(new_vecs)) )
    
    for key1, i1 in course_union_2id.items():
        v1 = new_vecs[i1]
        for key2, i2 in course_union_2id.items():
            v2 = new_vecs[i2]
            try:
                sim[i1][i2] = cosine_sim( v1, v2 )
            except Exception:
                print(v1, v2, Exception)
        if i1%20==0:
            print("%d rows iterated"%i1)
    
    with open("csim_matrix.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(sim) 

    pickle.dump(sim, open( "csim_matrix.pkl",'wb'), pickle.HIGHEST_PROTOCOL)
    print("similarity_train Ends")
    return None


def TryCosSimScheme():

    ''' One-off testing code '''

    texts = [
        "code",
        "algorithms",
        "literature",
        "language",
    ]
    vecs = [text2Vec(text) for text in texts]
    maxs = [max( [ vecs[j][i] for j in range(len(vecs))] ) for i in range(len(vecs[0])) ]
    mins = [min( [ vecs[j][i] for j in range(len(vecs))] ) for i in range(len(vecs[0])) ]
    means = [ np.mean( [ vecs[j][i] for j in range(len(vecs))] ) for i in range(len(vecs[0])) ]
    
    print(texts)
    print()

    print("vec")
    for vec1 in vecs:
        for vec2 in vecs:
            print( "%.3f"%(cosine_sim(vec1, vec2)), end='\t')
        print()
    print()

    print("vec-mean, 0 if <0")
    vecs2 = [ [vecs[j][i]-means[i] for i in range(len(vecs[0]))] for j in range(len(vecs))]
    for vec1 in vecs2:
        for vec2 in vecs2:
            print( "%.3f"%(max(cosine_sim(vec1, vec2),0)), end='\t')
        print()
    print()

    print("(vec-min)/(max-min)")
    vecs3 = [ [(vecs[j][i]-mins[i])/(maxs[i]-mins[i]) for i in range(len(vecs[0]))] for j in range(len(vecs))]
    for vec1 in vecs3:
        for vec2 in vecs3:
            print( "%.3f"%(cosine_sim(vec1, vec2)), end='\t')
        print()


def TestSim_irrelevant_corpus():
    '''
    Webtext: Wine
    '''
    wine = ' '.join(webtext.words('wine.txt'))
    tokens = tokenize(wine)
    tokens = [ tokens[i*100 :(i+1)*100 ] for i in range(int(len(tokens)/100))]
    vecs = [ text2Vec(' '.join(token))[0] for token in tokens ]
    print(len(vecs))
    print(vecs[0])
    # 不要scale. 直接和最naive的course description vectors比cosine similarity.

def TestStopWords():
    p = Paths()
    id_vec = dict()
    cinfo = pd.read_csv(p.cinfo_path, dialect="excel")
    
    name_id = pickle.load(open(p.nameid_path,'rb'))
    
    df = pd.DataFrame()
    c=0
    wc1 = 0
    wc2 = 0
    for i, row in cinfo.iterrows():
        key = row['Subject'] + row['Catalog']
        if i%20==0:
            print(i,"courses iterated (", key, ")")
        if key in name_id:
            cid = name_id[key]
            if cid not in id_vec:
                vec, c1, c2 = tokenize(row['Long Title']+' '+row['DESCRIPTION'], True)
                if vec == None:
                    # 這一步會再換一套id
                    print(key, vec)
                    continue
                wc1 += c1
                wc2 += c2
    print("Words used:", wc1, "of", wc2, "(", wc1/wc2, "%)")

if __name__=='__main__':
    #union_course()
    #print(len(words.manual))
    #id2Vec_train()
    scale_id_vec() #這就會存similarity了 不用再train
    #similarity_train()
    #CSim_pkl2csv()
    
    #TestSim_irrelevant_corpus()
    #TestStopWords()
