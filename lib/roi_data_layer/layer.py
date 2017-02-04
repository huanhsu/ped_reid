# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""The data layer used during training to train a Fast R-CNN network.

RoIDataLayer implements a Caffe Python layer.
"""

import caffe
from fast_rcnn.config import cfg
from roi_data_layer.minibatch import get_minibatch
import numpy as np
import yaml
from multiprocessing import Process, Queue
from random import randrange
import math
import pdb

class RoIDataLayer(caffe.Layer):
    """Fast R-CNN data layer used for training."""

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        if cfg.TRAIN.ASPECT_GROUPING:
            widths = np.array([r['width'] for r in self._roidb])
            heights = np.array([r['height'] for r in self._roidb])
            horz = (widths >= heights)
            vert = np.logical_not(horz)
            horz_inds = np.where(horz)[0]
            vert_inds = np.where(vert)[0]
            inds = np.hstack((
                np.random.permutation(horz_inds),
                np.random.permutation(vert_inds)))
            inds = np.reshape(inds, (-1, 2))
            row_perm = np.random.permutation(np.arange(inds.shape[0]))
            inds = np.reshape(inds[row_perm, :], (-1,))
            self._perm = inds
        else:
            self._perm = np.random.permutation(np.arange(len(self._roidb)))
        self._cur = 0

    #Huan add
    def _get_triplet_inds(self):
        """Return triplet set inds"""      
        anchor_inds = 0
        positive_inds = 0
        negative_inds = 0

        getAnchor = 0
        getPositive = 0
        getNegative = 0

        anchorPid = 0
        positivePid = 0
        negativePid = 0

        #lets find an anchor that does not have all gt_pids == [-1,-1... -1]
        for indexAnchordata in range(self._cur,len(self._roidb)):
            if getAnchor:
               anchor_inds = indexAnchordata-1
               break #jump out anchor data loop
            #for indexPid in range(len(self._roidb[self._perm[indexAnchordata]]['gt_pids'])):
            #    anchorPid =  self._roidb[self._perm[indexAnchordata]]['gt_pids'][indexPid] 
	    #    if anchorPid == 5532:
            #       continue
            #    else:
            #       anchor = self._roidb[self._perm[indexAnchordata]]
                   #choose the first one which is not -1 
            #       getAnchor = 1
            #       break #jump out anchor pids loop
            archorPid_list = np.where(self._roidb[self._perm[indexAnchordata]]['gt_pids']!=-1)
            if len(archorPid_list[0]) == 0:
               continue
            else:
               archorPid_index = randrange(0,len(archorPid_list[0]))
               archorPid_index = archorPid_list[0][archorPid_index]
               anchorPid = self._roidb[self._perm[indexAnchordata]]['gt_pids'][archorPid_index]
               getAnchor = 1
         
        #if got an anchor, then lets get positive sample from begin to end
        for indexPosdata in range(0,len(self._roidb)):
            if getPositive:
               positive_inds = indexPosdata-1
               break #jump out positive data loop
            elif indexPosdata == anchor_inds:
               continue #because searching from 0 to end, do not return anchor again

            for indexPid in range(len(self._roidb[self._perm[indexPosdata]]['gt_pids'])):
                positivePid =  self._roidb[self._perm[indexPosdata]]['gt_pids'][indexPid] 
	        if positivePid != anchorPid:
                   continue
                else:
                   positive = self._roidb[self._perm[indexPosdata]] 
                   getPositive = 1
                   break #jump out positive pids loop
        
        #pdb.set_trace()
        indexNegdata = 0
        #Get negative        
        while True:
            #print ('layer negative while')
            if getNegative:
               negative_inds = indexNegdata
               break #jump out negative data loop
            else:
               indexNegdata = randrange(0,len(self._perm))

            for indexPid in range(len(self._roidb[self._perm[indexNegdata]]['gt_pids'])):
                negativePid =  self._roidb[self._perm[indexNegdata]]['gt_pids'][indexPid] 
	        if negativePid == anchorPid or negativePid == -1 or negativePid == 5532:
                   continue
                else:
                   negative = self._roidb[self._perm[indexNegdata]] 
                   getNegative = 1
                   break #jump out negative pids loop
        
        #print ('Triplet set'+' an:'+str(anchorPid)+' '+'pos:'+str(positivePid)+' '+'neg:'+str(negativePid))
        return self._perm[anchor_inds], self._perm[positive_inds], self._perm[negative_inds], getPositive


    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        if self._cur + cfg.TRAIN.IMS_PER_BATCH >= len(self._roidb):
            self._shuffle_roidb_inds()
        #db_inds = self._perm[self._cur:self._cur + cfg.TRAIN.IMS_PER_BATCH]
        db_inds = np.zeros(cfg.TRAIN.IMS_PER_BATCH,dtype=np.int)
        #triplet loss
        while self._cur<len(self._roidb):
              #print('layer self._cur while')
              anchor_inds, positive_inds, negative_inds, getPositive= self._get_triplet_inds()
              if not getPositive: #do not have matching positive for anchor
                     self._cur += 1
              else:
                  
                  db_inds[0] = anchor_inds
                  db_inds[1] = positive_inds
                  db_inds[2] = negative_inds
                  self._cur = anchor_inds
                  break
        self._cur += 1 #go to next anchor
        return db_inds

    def _get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch.

        If cfg.TRAIN.USE_PREFETCH is True, then blobs will be computed in a
        separate process and made available through self._blob_queue.
        """
        if cfg.TRAIN.USE_PREFETCH:
            return self._blob_queue.get()
        else:
            db_inds = self._get_next_minibatch_inds()
            minibatch_db = [self._roidb[i] for i in db_inds]
            return get_minibatch(minibatch_db, self._num_classes)

    def set_roidb(self, roidb):
        """Set the roidb to be used by this layer during training."""
        self._roidb = roidb
        self._shuffle_roidb_inds()
        if cfg.TRAIN.USE_PREFETCH:
            self._blob_queue = Queue(10)
            self._prefetch_process = BlobFetcher(self._blob_queue,
                                                 self._roidb,
                                                 self._num_classes)
            self._prefetch_process.start()
            # Terminate the child process when the parent exists
            def cleanup():
                print 'Terminating BlobFetcher'
                self._prefetch_process.terminate()
                self._prefetch_process.join()
            import atexit
            atexit.register(cleanup)

    def setup(self, bottom, top):
        """Setup the RoIDataLayer."""

        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str_)

        self._num_classes = layer_params['num_classes']

        self._name_to_top_map = {}

        # data blob: holds a batch of N images, each with 3 channels
        idx = 0
        top[idx].reshape(cfg.TRAIN.IMS_PER_BATCH, 3,
            max(cfg.TRAIN.SCALES), cfg.TRAIN.MAX_SIZE)
        self._name_to_top_map['data'] = idx
        idx += 1

        if cfg.TRAIN.HAS_RPN:
            top[idx].reshape(1, 3)
            self._name_to_top_map['im_info'] = idx
            idx += 1

            top[idx].reshape(1, 4)
            self._name_to_top_map['gt_boxes'] = idx
            idx += 1

            #add for IMS.BATCH.NUMBER
            top[idx].reshape(1)
            self._name_to_top_map['im_inds'] = idx
            idx += 1
        else: # not using RPN
            # rois blob: holds R regions of interest, each is a 5-tuple
            # (n, x1, y1, x2, y2) specifying an image batch index n and a
            # rectangle (x1, y1, x2, y2)
            top[idx].reshape(1, 5)
            self._name_to_top_map['rois'] = idx
            idx += 1

            # labels blob: R categorical labels in [0, ..., K] for K foreground
            # classes plus background
            top[idx].reshape(1)
            self._name_to_top_map['labels'] = idx
            idx += 1

            if cfg.TRAIN.BBOX_REG:
                # bbox_targets blob: R bounding-box regression targets with 4
                # targets per class
                top[idx].reshape(1, self._num_classes * 4)
                self._name_to_top_map['bbox_targets'] = idx
                idx += 1

                # bbox_inside_weights blob: At most 4 targets per roi are active;
                # thisbinary vector sepcifies the subset of active targets
                top[idx].reshape(1, self._num_classes * 4)
                self._name_to_top_map['bbox_inside_weights'] = idx
                idx += 1

                top[idx].reshape(1, self._num_classes * 4)
                self._name_to_top_map['bbox_outside_weights'] = idx
                idx += 1

        print 'RoiDataLayer: name_to_top:', self._name_to_top_map
        assert len(top) == len(self._name_to_top_map)

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch()
        for blob_name, blob in blobs.iteritems():
            top_ind = self._name_to_top_map[blob_name]
            # Reshape net's input blobs
            top[top_ind].reshape(*(blob.shape))
            # Copy data into net's input blobs
            top[top_ind].data[...] = blob.astype(np.float32, copy=False)

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

class BlobFetcher(Process):
    """Experimental class for prefetching blobs in a separate process."""
    def __init__(self, queue, roidb, num_classes):
        super(BlobFetcher, self).__init__()
        self._queue = queue
        self._roidb = roidb
        self._num_classes = num_classes
        self._perm = None
        self._cur = 0
        self._shuffle_roidb_inds()
        # fix the random seed for reproducibility
        np.random.seed(cfg.RNG_SEED)

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        # TODO(rbg): remove duplicated code
        self._perm = np.random.permutation(np.arange(len(self._roidb)))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        # TODO(rbg): remove duplicated code
        if self._cur + cfg.TRAIN.IMS_PER_BATCH >= len(self._roidb):
            self._shuffle_roidb_inds()

        db_inds = self._perm[self._cur:self._cur + cfg.TRAIN.IMS_PER_BATCH]
        self._cur += cfg.TRAIN.IMS_PER_BATCH
        return db_inds

    def run(self):
        print 'BlobFetcher started'
        while True:
            db_inds = self._get_next_minibatch_inds()
            minibatch_db = [self._roidb[i] for i in db_inds]
            blobs = get_minibatch(minibatch_db, self._num_classes)
            self._queue.put(blobs)
