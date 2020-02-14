#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 16:24:46 2018

Recode the SSimilarity 
(because of the chaos in the original one)

@author: shanglinghsu
"""
#from CSimilarity import Paths
#import matplotlib.pyplot as plt
#from sklearn.manifold import TSNE
import numpy as np
from scipy import sparse
#from math import sqrt,log
import pickle
import pandas as pd
import time


########################################################################
# Read in the Course Name <-> Index Table of Course Simialrity Matrix
    # How many entries? 905.
    # Do the indices really match? Yes!
########################################################################
def loadCourseIdConverter():

    # cname to csimId
    cinfo = pd.read_csv("course_id-453777-nltk-stop.csv")
    cname2csimId = dict(zip(cinfo['Subject']+cinfo['Catalog'], cinfo['ID']))
    
    # sampleId
    sample_index2cname = pickle.load(open('./dataset/courses.pkl','rb'))
    sample = pickle.load(open('./dataset/sample.pkl','rb'))
    
    # sampleId to csimId
    sampleId2csimId = []
    for i in range(len(sample_index2cname)):
        if sample_index2cname[i] in cname2csimId:
            sampleId2csimId.append(cname2csimId[sample_index2cname[i]])
        else:
            sampleId2csimId.append(-1)
    return sample, sampleId2csimId 


def CalculateNewSample():    
    sample, sampleId2csimId = loadCourseIdConverter()
                       
    # New samples with consistent course id
    row = []
    col = []
    data = []
    for sid, a_students_grade in enumerate(sample):
        for cid, (grade, semester) in enumerate(a_students_grade):
            if semester >= 1630 or grade < 0 or sampleId2csimId[cid] < 0:
                continue
            row.append(sid)
            col.append(sampleId2csimId[cid])
            data.append(grade+10.0) # To distinguish it from 0.0 in sparse

    new_sample = sparse.coo_matrix((data, (row, col)))
    print("new_sample := sparse matrix of shape {} with {} data".format(
            new_sample.get_shape(),new_sample.getnnz()))
    return new_sample


########################################################################
# Calculate Students' stud_x_mean
    # student x
    # Consider only valid courses (the ones with name in 905 Courses)
########################################################################   
def CalculateStudentsMean():
    new_sample = CalculateNewSample()
    NUM_STUD, NUM_COUR = new_sample.get_shape()
    stud_means = np.zeros(NUM_STUD)
    for i in range(NUM_STUD):
        if i%50 == 0:
            print('{:>02}:{:>02}:{:>02} | {:>3} ({:>5.2f}%) iterated.'.format(
                    time.localtime()[3],time.localtime()[4],time.localtime()[5],
                    i, i/NUM_STUD*100))
        stud_row = new_sample.getrow(i)
        stud_sum = 0.0
        for j in range(NUM_COUR):
            grade_plus_10 = stud_row.getcol(j).toarray()[0][0]
            if grade_plus_10 < 1.0:
                continue
            grade = grade_plus_10-10.0
            stud_sum += grade
        stud_means[i] = stud_sum / np.max([stud_row.getnnz(), 1.0])
    df = pd.DataFrame()
    df['means'] = stud_means
    df.to_csv(path_or_buf='_new_stud_sample_mean.csv', index=False)
    return stud_means

def GetStudentsMean():
    return pd.read_csv('_new_stud_sample_mean.csv')



########################################################################
# Calculate Students' (∑(stud_x,i - stud_x_mean)^2)^.5
    # i iterates the courses student x took
    # Consider only valid courses (the ones with name in 905 Courses)
########################################################################  
def CalculateStudentsRMSE():
    new_sample = CalculateNewSample()
    NUM_STUD, NUM_COUR = new_sample.get_shape()
    
    stud_means = GetStudentsMean()
    stud_rmses = np.zeros(NUM_STUD)
    
    for stud, mean in enumerate(stud_means['means']):
        if stud%50 == 0:
            print('{:>02}:{:>02}:{:>02} | {:>3} ({:>5.2f}%) iterated.'.format(
                    time.localtime()[3],time.localtime()[4],time.localtime()[5],
                    stud, stud/NUM_STUD*100))
        sum_of_squares = 0.0
        stud_row = new_sample.getrow(stud)
        for j in range(NUM_COUR):
            grade_plus_10 = stud_row.getcol(j).toarray()[0][0]
            if grade_plus_10 < 1.0:
                continue
            grade = grade_plus_10-10.0
            sum_of_squares += (grade-mean)**2
        stud_rmses[stud] = np.sqrt(
                sum_of_squares / np.max([stud_row.getnnz(), 1.0]))
        
    stud_means['rmse'] = stud_rmses
    stud_means.to_csv(path_or_buf='_new_stud_sample_mean_rmse.csv', index=False)
    return stud_means

def GetStudentsMeanAndRmse():
    return pd.read_csv('_new_stud_sample_mean_rmse.csv')



########################################################################  
# Calculate Students' Simialrity Matrix
    # Read in the Course Simialrity Matrix
    # Iterate pairs of students
    # Dump the matrix in csv with metadata
########################################################################  
    
def CalculateStudentSimialrity(NUM_SAMPLE):
    
    """ 
    NUM_SAMPLE: >=1032 to iterate all pairs of students
    """
    
    # Coure Simialrity Matrix: convert from dataframe to np array
    courseSimilarity = pd.read_csv(
            "scaled_csim_matrix-nltk-stop-ordinary.csv").drop(
                    columns=['SubjectCatalog']).values
    
    new_sample = CalculateNewSample()    
    NUM_STUD, NUM_COUR = new_sample.get_shape()
    
    mean_and_rmse = GetStudentsMeanAndRmse()    
    studentSimilarity = np.zeros((NUM_STUD,NUM_STUD))
    
    for stud_x in range(NUM_STUD):
        if stud_x >= NUM_SAMPLE:
            break
        
        print('{:>02}:{:>02}:{:>02} | {:>3} ({:>5.2f}%) iterated.'.format(
                time.localtime()[3],time.localtime()[4],time.localtime()[5],
                stud_x, stud_x/NUM_STUD*100))
        
        x_mean = mean_and_rmse['means'][stud_x]
        x_rmse = mean_and_rmse['rmse'][stud_x]
        stud_x_row = new_sample.getrow(stud_x)
        _, x_cours = stud_x_row.nonzero()
         
        for stud_y in range(NUM_STUD):
            if stud_y >= NUM_SAMPLE:
                break
            y_mean = mean_and_rmse['means'][stud_y]
            y_rmse = mean_and_rmse['rmse'][stud_y]
            stud_y_row = new_sample.getrow(stud_y)
            _, y_cours = stud_y_row.nonzero()
            
            sum_of_sum = 0.0
            weights = 0.0
                
            for x_cour in x_cours:
                x_grade_plus_10 = stud_x_row.getcol(x_cour).toarray()[0][0]
                if x_grade_plus_10  < 1.0:
                    continue
                x_grade = x_grade_plus_10-10.0
                
                
                for y_cour in y_cours:
                    y_grade_plus_10 = stud_y_row.getcol(y_cour).toarray()[0][0]
                    if y_grade_plus_10  < 1.0:
                        continue
                    y_grade = y_grade_plus_10-10.0
                    
                    weight = courseSimilarity[x_cour][y_cour]
                    sum_of_sum += weight*(x_grade-x_mean)*(y_grade-y_mean)
                    weights += np.abs(weight)
        
            ### WARNING: Some of the values are NaN ###
            studentSimilarity[stud_x][stud_y] = sum_of_sum/(x_rmse*y_rmse*weights)
            
    for x in range(NUM_SAMPLE):
        for y in range(NUM_SAMPLE):
            # Do not use the -1.0 < sim < 1.0 constraint
            studentSimilarity[x][y] = studentSimilarity[x][y]/np.sqrt(studentSimilarity[x][x]*studentSimilarity[y][y]),
                    
    np.savetxt("_SSim.csv", studentSimilarity, delimiter=",")
    return studentSimilarity
    
def GetStudentSimilariy():
    return pd.read_csv('_SSim.csv', header=None)


if __name__=='__main__':
    CalculateStudentsMean()
    CalculateStudentsRMSE()
    CalculateStudentSimialrity(1032)
    #pass