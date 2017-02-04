# --------------------------------------------------------
# TRIPLET LOSS
# Copyright (c) 2015 Pinguo Tech.
# Written by David Lu
# --------------------------------------------------------

"""The data layer used during training a VGG_FACE network by triplet loss.
   The layer combines the input image into triplet.Priority select the semi-hard samples
"""
import caffe
import numpy as np
from numpy import *
import yaml
from multiprocessing import Process, Queue
from caffe._caffe import RawBlobVec
from sklearn import preprocessing
from random import randrange
import math
import config
import pdb

class TripletSelectLayer(caffe.Layer):
        
    def setup(self, bottom, top):
        """Setup the TripletSelectLayer."""
        self.triplet = config.BATCH_SIZE/3
        top[0].reshape(self.triplet,shape(bottom[0].data)[1])
        top[1].reshape(self.triplet,shape(bottom[0].data)[1])
        top[2].reshape(self.triplet,shape(bottom[0].data)[1])

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        top_archor = []
        top_positive = []
        top_negative = []
        labels = []
        self.tripletlist = []
        self.no_residual_list=[]
        aps = {}
        ans = {}
        #collect triplet set
        imgs_idx = shape(bottom[1].data)[0] #3
        feats_idx = shape(bottom[1].data)[1] #64

        #archor, take from bottom[1].data[0]
        archor_list = np.intersect1d(np.where(bottom[1].data[0][0:feats_idx-1]!=5532), np.where(bottom[1].data[0][0:feats_idx-1]!=-1))
        if len(archor_list) == 0:
            archor_list = np.where(bottom[1].data[0][0:feats_idx-1]==5532)
            print('*** background as anchor ***')
        else:
            archor_list = np.where(bottom[1].data[0][0:feats_idx-1]!=5532)

        count = 0
        while True:
              archor_index = randrange(0,shape(archor_list)[1])
              archor_index = archor_list[0][archor_index]
              archor_value = bottom[1].data[0][archor_index]
              if archor_value<0 and count < 1000:
                 count += 1
                 continue
              pos_list_temp = np.where(bottom[1].data[1][0:feats_idx-1]==archor_value)
              if shape(pos_list_temp)[1]>0:
                 break

        #positive, take from bottom[1].data[1]
        positive_list = np.where(bottom[1].data[1][0:feats_idx-1]==archor_value)
        positive_index = randrange(0, shape(positive_list)[1])
        positive_index = positive_list[0][positive_index]
        positive_value = bottom[1].data[1][positive_index]
        
        #negative, take from bottom[1].data[2]
        negative_list = np.intersect1d(np.where(bottom[1].data[2][0:feats_idx-1]!=5532), np.where(bottom[1].data[2][0:feats_idx-1]!=-1))
        if len(negative_list) == 0:
           negative_list = np.where(bottom[1].data[2][0:feats_idx-1]==5532)
           print ('*** background as negative ***')
           negative_index = randrange(0, shape(negative_list)[0])
           negative_index = negative_list[0][negative_index]
        else:
           negative_index = randrange(0, shape(negative_list)[0])
           negative_index = negative_list[negative_index]
        negative_value = bottom[1].data[2][negative_index]

        print ('Triplet set'+' an:'+str(archor_value)+' '+'pos:'+str(positive_value)+' '+'neg:'+str(negative_value))
        #index in feature
        archor_feature_index = archor_index
        positive_feature_index = positive_index + feats_idx
        negative_feature_index = negative_index + feats_idx*2 

        #collect triplet set old
        archor_feature = bottom[0].data[archor_feature_index]
        for i in range(self.triplet):
            positive_feature = bottom[0].data[positive_feature_index]
            a_p = archor_feature - positive_feature
            ap = np.dot(a_p,a_p)
            aps[i+self.triplet] = ap
        aps = sorted(aps.items(), key = lambda d: d[1], reverse = True)
        for i in range(self.triplet):
            negative_feature = bottom[0].data[negative_feature_index]
            a_n = archor_feature - negative_feature
            an = np.dot(a_n,a_n)
            ans[i+self.triplet*2] = an
        ans = sorted(ans.items(), key = lambda d: d[1], reverse = True)  

        for i in range(self.triplet): 
            top_archor.append(bottom[0].data[archor_feature_index])
            top_positive.append(bottom[0].data[positive_feature_index])
            top_negative.append(bottom[0].data[negative_feature_index])
            if aps[i][1] >= ans[i][1]:
               self.no_residual_list.append(i)
            self.tripletlist.append([archor_feature_index,positive_feature_index,negative_feature_index])

        top[0].data[...] = np.array(top_archor).astype(float32)
        top[1].data[...] = np.array(top_positive).astype(float32)
        top[2].data[...] = np.array(top_negative).astype(float32)
    

    def backward(self, top, propagate_down, bottom):
        
        for i in range(len(self.tripletlist)):
            if not i in self.no_residual_list:
                bottom[0].diff[self.tripletlist[i][0]] = top[0].diff[i]
                bottom[0].diff[self.tripletlist[i][1]] = top[1].diff[i]
                bottom[0].diff[self.tripletlist[i][2]] = top[2].diff[i]
            else:
                bottom[0].diff[self.tripletlist[i][0]] = np.zeros(shape(top[0].diff[i]))
                bottom[0].diff[self.tripletlist[i][1]] = np.zeros(shape(top[1].diff[i]))
                bottom[0].diff[self.tripletlist[i][2]] = np.zeros(shape(top[2].diff[i]))

        #print 'backward-no_re:',bottom[0].diff[0][0]
        #print 'tripletlist:',self.no_residual_list

        

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass







