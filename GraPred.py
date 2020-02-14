import redo_SSim as ss
#import matplotlib.pyplot as plt
#from sklearn.manifold import TSNE
import numpy as np
from math import sqrt,log
#import pickle
import time
import pandas as pd
from scipy import stats

def testpara(a,b):
#if __name__ == '__main__':
    
    sample, sampleId2csimId = ss.loadCourseIdConverter()
    studsim = ss.GetStudentSimilariy()
    smean = ss.GetStudentsMean()['means']    


    def predict(sid1, cid):

        cid = sampleId2csimId[cid]
        cvec = [row[cid] for row in sample]    # grade vector

        all_fail = True
        sum_vote = 0.0
        sum_weight = 0.0
        num_record = 0
        total_record = 0
        
        for sid2, gt in enumerate(cvec):
            grade, term = gt
            if term==0 or term >= 1630:
                continue
            
            total_record += 1
            weight = studsim[sid1][sid2]
            if abs(weight) < a:        # 要算入的threshold
                continue 
    
            if grade > 0.5:
                all_fail = False
            else:
                continue                # 不算fail
            sum_vote += weight * (grade - smean[sid2])
            sum_weight += abs(weight)
            num_record += 1 
            
        if num_record < b:    #### This parameter matters!!! 83
            return -2.0, 0
        if all_fail:
            return -3.0, 0
        
        return max(min(smean[sid1] + sum_vote / sum_weight, 4.3),0), num_record/total_record

    # Baseline Comp
    SEs_pred= []
    SEs_bali= []

    num_pred = 0
    li_pred = list()
    SE = 0.0
    p_or_f = 0
    cold_start = 0
    sum_fracs = 0.0
    for sid1, s1 in enumerate(sample):
        for cid1, (grade, term) in enumerate(s1):
            if term==0 or term < 1630:
                continue
            
            # Baseline
            SEs_bali.append(abs(smean[sid1]-grade))
            
            pred, frac = predict(sid1,cid1)
            if pred > 0.3:
                num_pred += 1
                sum_fracs += frac
                li_pred.append( (sid1,cid1,term,grade,pred) )
                SE += (pred - grade) ** 2
                
                # Baseline
                SEs_pred.append(abs(pred - grade))
                
                """
                if num_pred % 300 == 0:
                    print("{:>2}:{:>2}:{:>2} | {:>4}: p-g= {:>3.2f} - {:>2.1f} = {:>3.2f}".format(
                        time.localtime()[3],time.localtime()[4],time.localtime()[5],
                        num_pred,pred,grade,pred-grade))
                """
            elif pred == -2:
                cold_start += 1
            elif pred == -3:
                p_or_f += 1
    
    try:
        RMSE = sqrt(SE/num_pred)
        MFracs = sum_fracs/num_pred
    except ZeroDivisionError:
        return 0.0, [], []
    
    print("a={:>3.2f}, b={}".format(a,b))
    print("RMSE={:>.6f}".format(RMSE))
    print("# pred:\t\t{}".format(num_pred))
    print("average % peer data used: {:>.6f}".format(MFracs))
    #print("% pred:\t\t{:>.6f}".format(num_pred/(cold_start+p_or_f)))
    #print("cold-start:\t{}".format(cold_start))
    #print("P-or-F:\t\t{}".format(p_or_f))
    
    
    return RMSE, SEs_pred, SEs_bali
    
    """
    df = pd.DataFrame()    
    df['Student'] = [row[0] for row in li_pred]
    df['Course'] = [row[1] for row in li_pred]
    df['Term'] = [row[2] for row in li_pred]
    df['Grade'] = [row[3] for row in li_pred]
    df['Prediction'] = [row[4] for row in li_pred]
    df.to_csv(path_or_buf='_predicted.csv', index=False)
    """
    
    
   
if __name__=='__main__':
    """
    X = 12
    Y = 1
    results = np.zeros((X,Y))
    for _a in range(X):
        a = 1.0/X*_a
        #print(a,end=' ')
        for _b in range(Y):
            #b = int(140/Y*_b)+2
            b = 25
            #print(b,end=' ')
            results[_a][_b] = testpara(a,b)
            print("({:>.2f},{:>3}):{:>.6f}".format(a,b,results[_a][_b]), end='\n')
        print()
    """
    
    RMSE, SEs_pred, SEs_bali = testpara(0.33,25)
    s,p = scipy.stats.ttest_ind(SEs_bali, SEs_pred, equal_var=False)
    print(s,p)