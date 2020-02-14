from CSimilarity import Paths
#from gensim.models import word2vec
import matplotlib.pyplot as plt
#from sklearn.manifold import TSNE
import numpy as np
#from math import sqrt,log
import pickle
import pandas as pd
import time

def PlotsStuddist(n):
    p = Paths()
    #studsim = pickle.load(open(p.studsim, 'rb'))
    #studsim = pickle.load(open(p.studsim_2, 'rb'))
    studsim = pickle.load(open("studsim-453777-nltk-stop.pkl", 'rb'))
    studsimdist = np.zeros(n+1)
    dmin=-1.0
    dmax=1.0
    drange=dmax-dmin
    width = drange/n
    
    for it1, rows in enumerate(studsim):
        # if it1==1000:####################
        #     break
        for it2, sim in enumerate(rows):
            # if it2==1000:####################
            #     break
            if it1==it2:
                print(sim)
                break
            if sim>dmin:
                studsimdist[ int((sim-dmin)/width) ] += 1
            else:
                print(sim)
    
    # Plot the dist
    fig, ax = plt.subplots()
    index = list( dmin+i*width for i in range(0,n+1))
    dist = list( studsimdist[i]/2  for i in range(0,n+1))
    bar_width = 0.003
    opacity = 0.4

    ax.bar(index, dist, bar_width, alpha=opacity, color='b')
    ax.set_xlabel('Similarity')
    ax.set_ylabel('Number of Students')
    ax.set_title('Distribution of Similarity Among Students')
    ax.legend()
    fig.tight_layout()
    fig.savefig('plot/studsim_dist_453777_original.png', dpi=120)
    plt.show()

def cid_converter():

    """ 
    This convert the Course ID used in 'sample', 'smean', ...
    (i.e., from Grades)
    to the Course ID used in csim, course_id*
    (i.e., from CInfo, or Course Descriptions)
    """

    #print(id2c[105]) #'COMP2711', sample, smean, sgsq用的id
    id2c = pickle.load(open(Paths().courses,'rb'))

    #csim用的id
    cid_in_csim = pd.read_csv("course_id-453777-nltk-stop.csv")
    #cname = id2c[sample_cid]
    print(len(id2c),len(cid_in_csim))
    
    new_cids = [-1 for i in range(len(id2c))]
    for old_id, cname in enumerate(id2c):
        if old_id%10==0:
            print(old_id, "rows iterated")
        for new_cid, row in cid_in_csim.iterrows():
            if row['Subject']+row['Catalog'] == cname:
                new_cids[old_id] = new_cid
    
    pickle.dump(new_cids, open("sample_cid_2_csim_cid.pkl",'wb'), pickle.HIGHEST_PROTOCOL)
    print(new_cids[:20])

def StudentSimilarity():
    
    print("StudentSimilarity Starts")
    
    p = Paths()
    sample = pickle.load(open(p.sample, 'rb'))
    smean = pickle.load(open(p.smean, 'rb'))
    sgsq = pickle.load(open(p.sgsq, 'rb'))
    #sim = pickle.load(open(p.resim_path, 'rb'))
    sim = pickle.load(open("scaled_csim_matrix-nltk-stop-ordinary.pkl",'rb'))
    cid_converter = pickle.load(open("sample_cid_2_csim_cid.pkl", 'rb'))

    studsim = np.zeros((len(sample),len(sample)))
    num_samples = len(sample)

    # student 1
    for s1, l1 in enumerate(sample):    
        if s1%1==0:
            print('{:>02}:{:>02}:{:>02} | {:>3} samples of {} ({}%) iterated.'.format(
                time.localtime()[3],time.localtime()[4],time.localtime()[5],s1,
                num_samples, s1/num_samples*100))
        
        # if s1==1000:####################
        #     return

        # student 2
        for s2, l2 in enumerate(sample):    
            
            ''' Calculate similarity between student 1 & 2 '''
            
            # if s2==1000:####################
            #     break

            # course 1
            sum_weight = 0.0    # ∑∑ |W(i,j)|
            for c1, (g1, t1)  in enumerate(l1):    
                if t1==0 or t1>=1630:
                    continue
                c1 = cid_converter[c1]
                if c1==-1:
                    continue

                # course 2
                for c2, (g2, t2) in enumerate(l2): 
                    if t2==0 or t2>=1630:
                        continue
                    c2 = cid_converter[c2]
                    if c2==-1:
                        continue
                    
                    a = sim[c1][c2]
                    a = 1
                    b = g1-smean[s1]
                    c = g2-smean[s2]
                    studsim[s1][s2] += a*b*c
                    sum_weight += abs(a)
                    #print(abs(a), int(a*b*c*10000)/10000)
                    

            # Save in studism    
            studsim[s1][s2] /= (sum_weight * sgsq[s1] * sgsq[s2])
            #studsim[s1][s2] /= (sgsq[s1] * sgsq[s2])
    
    # Dump studism
    pickle.dump(studsim, open("studsim-453777-nltk-stop.pkl",'wb'), pickle.HIGHEST_PROTOCOL)
    print("StudentSimilarity Ends")
        
def DumpSampleMeanSquare():
    p = Paths()
    sample = pickle.load(open(p.sample, 'rb'))
    smean = np.zeros(len(sample))
    #smean = pickle.load(open(p.smean, 'rb'))
    sgsq = np.zeros(len(sample))
    
    for i, li in enumerate(sample):
        c=0
        for (g,t) in li:
            if t!=0 and t<1630:
                sgsq[i] += (g-smean[i])**2
                #smean[i] += g
                #c+=1
        #smean[i]/=c
        
    sqrtsgsq = list( sqrt(i) for i in sgsq )
    pickle.dump(sqrtsgsq, open( p.sgsq,'wb'), pickle.HIGHEST_PROTOCOL)
    #pickle.dump(smean,open( p.smean,'wb'), pickle.HIGHEST_PROTOCOL)



if __name__ == '__main__':
    #cid_converter()
    StudentSimilarity()
    #PlotsStuddist(30)
    #DumpSampleMeanSquare()
    
    ''' visualize the student similarity result is complex
    Example: https://plot.ly/scikit-learn/plot-stock-market/#plot-results
    '''