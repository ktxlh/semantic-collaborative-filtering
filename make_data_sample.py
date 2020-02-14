'''

    To make the data in desired format and take random sample from it.

    * real term: flat[student_id][course_id] = (grade, term_id)
        * Extract the following from this matrix

    * (sampled) sample[student_id][course_id] = (grade, student_term)
        * To trace the courses the student takes beforehand
        * Terms reserved for testing: 1630 (& 1640?)
        * similarity: student_sim[s1_id][s2_id]

'''
from CSimilarity import CourseSimilarity, Paths
from test_GDist import l2f

from gensim.models import word2vec
import numpy as np
#from scipy import sparse
import random
import pandas as pd

import pickle

gdata = pd.read_csv(Paths().grade_path, dialect="excel")

def dump():
    p = Paths()
    #t2id = dict()   # masked student id to index
    #trms = list()   # sorted list of masked student id
    #s2id = dict()   # masked student id to index
    #stds = list()   # sorted list of masked student id
    c2id = dict()   # course id to index
    crss = list()   # sorted list of course id
    
    total = len(gdata)

    print("dump: Start iteraton")
    for i, row in gdata.iterrows():
        if i%10000==0:
            print("{} of {} rows interated ({}%)".format(i,total,i/total*100))
        osid = row['Masked ID']    # original student id
        #ocid = row['Course ID']    # original course id
        ocid = row['Subject'] + row['Catalog']
        otid = row['Enroll Term']
        #if osid not in s2id:
        #    s2id[osid] = len(stds)
        #    stds.append(osid)
        if ocid not in c2id:
            c2id[ocid] = len(crss)
            crss.append(ocid)
        #if otid not in t2id:
        #    t2id[otid] = len(trms)
        #    trms.append(otid)
    print("dump: End iteraton")
    #pickle.dump(t2id, open(p.term2id,'wb'), pickle.HIGHEST_PROTOCOL)
    #pickle.dump(trms, open(p.terms,'wb'), pickle.HIGHEST_PROTOCOL)
    #pickle.dump(s2id, open(p.student2id,'wb'), pickle.HIGHEST_PROTOCOL)
    #pickle.dump(stds, open(p.students,'wb'), pickle.HIGHEST_PROTOCOL)
    pickle.dump(c2id, open(p.course2id,'wb'), pickle.HIGHEST_PROTOCOL)
    pickle.dump(crss, open(p.courses,'wb'), pickle.HIGHEST_PROTOCOL)
    print("Dump job finished.")

def csvToMatrix():
    
    '''  Read in CSV, dump the matrice '''
    
    print('csvToMatrix Starts')
    p=Paths()
    s2id = pickle.load(open(p.student2id, 'rb'))
    c2id = pickle.load(open(p.course2id, 'rb'))
    t2id = pickle.load(open(p.term2id, 'rb'))
    #grades = list( list( list(-1.0 for k in range(len(c2id)) )
    #                               for j in range(len(t2id)) )
    #                               for i in range(len(s2id)) )
    flat = list(list( (-1.0,0) for j in range(len(c2id))) for i in range(len(s2id)))

    print("csvToMatrix: Start iteraton")
    print("len(c2id)={}".format(len(c2id)))
    not_in = 0
    for i, row in gdata.iterrows():
        if i%10000==0:
            print("{} rows interated".format(i))

        g = l2f().convert(row['Grade'])
        if g >= 0:
            ocid = row['Subject'] + row['Catalog']
            if ocid in c2id:
                s = s2id[ row['Masked ID'] ]
                term = row['Enroll Term']
                c = c2id[ ocid ]
                #grades[s][t][c]= g
                flat[s][c] = (g,term)
            else:
                not_in += 1
                print(ocid)

    print("csvToMatrix: End iteraton: not_in={}".format(not_in))
    
    #pickle.dump(grades, open(p.grades,'wb'), pickle.HIGHEST_PROTOCOL)
    pickle.dump(flat,   open(p.flat,'wb'), pickle.HIGHEST_PROTOCOL)
    print("Matrice dumped")
    print('csvToMatrix Ends')

def make_sample():
    print("make_sample starts")
    p = Paths()
    flat = pickle.load(open(p.flat, 'rb'))
    stds = pickle.load(open(p.students, 'rb'))
    s2id = pickle.load(open(p.student2id, 'rb'))

    #print(len(flat), len(flat[0]), type(flat),type(flat[0]))
    
    rand_smpl = random.sample(stds, int(len(stds)/11))
    print(len(rand_smpl))

    sample = list()
    for s in rand_smpl:
        sample.append(flat[ s2id[s] ])

    pickle.dump(sample, open(p.sample,'wb'), pickle.HIGHEST_PROTOCOL)
    print("make_sample ends")

if __name__ == '__main__':
    #dump()
    #csvToMatrix()
    make_sample()