#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch

from torch.utils.data import Dataset

class TrainDataset(Dataset):
    def __init__(self, triples, nentity, nrelation, negative_sample_size, mode, ent_conc, nconcept,rel_h,rel_t):
        self.len = len(triples)
        self.triples = triples
        self.triple_set = set(triples)
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        self.count = self.count_frequency(triples)
        self.true_head, self.true_tail = self.get_true_head_and_tail(self.triples)

        self.ent_conc = ent_conc  # entity to concept
        self.nconcept = nconcept  # num of concepts for padding with num
        self.rel_h = rel_h
        self.rel_t = rel_t
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        positive_sample = self.triples[idx]

        head, relation, tail = positive_sample

        subsampling_weight = self.count[(head, relation)] + self.count[(tail, -relation-1)]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))
        
        negative_sample_list = []
        negative_sample_size = 0

        while negative_sample_size < self.negative_sample_size:
            negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size*2)
            if self.mode == 'head-batch':
                mask = np.in1d(
                    negative_sample, 
                    self.true_head[(relation, tail)], 
                    assume_unique=True, 
                    invert=True
                )
            elif self.mode == 'tail-batch':
                mask = np.in1d(
                    negative_sample, 
                    self.true_tail[(head, relation)], 
                    assume_unique=True, 
                    invert=True
                )
            else:
                raise ValueError('Training batch mode %s not supported' % self.mode)
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size
        
        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]

        negative_sample = torch.LongTensor(negative_sample)

        positive_sample = torch.LongTensor(positive_sample)

        h_concept = []
        if str(head) in self.ent_conc:
            h_concept = self.ent_conc[str(head)]
        elif str(relation) in self.rel_h:
            h_concept = self.rel_h[str(relation)]
        h_concept = [x - 1 for x in h_concept]
        h_c = [0] * self.nconcept
        for i in h_concept:
            h_c[i] = 1
        h_concept = torch.FloatTensor(h_c)

        t_concept = []
        if str(tail) in self.ent_conc:
            t_concept = self.ent_conc[str(tail)]
        elif str(relation) in self.rel_t:
            t_concept = self.rel_t[str(relation)]
        t_concept = [x - 1 for x in t_concept]
        t_c = [0] * self.nconcept
        for i in t_concept:
            t_c[i] = 1
        t_concept = torch.FloatTensor(t_c)
            
        return positive_sample, negative_sample, subsampling_weight, self.mode, h_concept, t_concept
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        mode = data[0][3]
        h_concept = torch.stack([_[4] for _ in data], dim=0)
        t_concept = torch.stack([_[5] for _ in data], dim=0)
        return positive_sample, negative_sample, subsample_weight, mode, h_concept, t_concept
    
    @staticmethod
    def count_frequency(triples, start=4):
        '''
        Get frequency of a partial triple like (head, relation) or (relation, tail)
        The frequency will be used for subsampling like word2vec
        '''
        count = {}
        for head, relation, tail in triples:
            if (head, relation) not in count:
                count[(head, relation)] = start
            else:
                count[(head, relation)] += 1

            if (tail, -relation-1) not in count:
                count[(tail, -relation-1)] = start
            else:
                count[(tail, -relation-1)] += 1
        return count
    
    @staticmethod
    def get_true_head_and_tail(triples):
        '''
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        '''
        
        true_head = {}
        true_tail = {}

        for head, relation, tail in triples:
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = []
            true_tail[(head, relation)].append(tail)
            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = []
            true_head[(relation, tail)].append(head)

        for relation, tail in true_head:
            true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
        for head, relation in true_tail:
            true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))                 

        return true_head, true_tail


class TestDataset(Dataset):
    def __init__(self, triples, all_true_triples, nentity, nrelation, mode,  rel_h, rel_t, ent_conc, is_filter=False):
        self.len = len(triples)
        self.triple_set = set(all_true_triples)
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.mode = mode
        self.rel_h = rel_h
        self.rel_t = rel_t
        self.ent_conc = ent_conc
        self.is_filter = is_filter

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]
        set_rel2h = set()
        if str(relation) in self.rel_h:
            set_rel2h = set(self.rel_h[str(relation)])

        set_rel2t = set()
        if str(relation) in self.rel_t:
            set_rel2t = set(self.rel_t[str(relation)])


        if self.mode == 'head-batch':
            if(self.is_filter):

                # tmp = [(0, rand_head) if (rand_head, relation, tail) not in self.triple_set
                #     else (-1, head) for rand_head in range(self.nentity)]
                # tmp[head] = (0, head)
                tmp = []
                for rand_head in range(self.nentity):
                    if str(rand_head) in self.ent_conc:  # entity has no concept
                        set_head2conc = set(self.ent_conc[str(rand_head)])
                    else:  # entity has concept(s)
                        set_head2conc = set_rel2h
                    if (rand_head, relation, tail) not in self.triple_set:
                        if len(set_rel2h & set_head2conc) > 0:
                            tmp.append((0, rand_head))
                        else:
                            tmp.append((-1, head))
                    else:
                        tmp.append((-1, head))
                tmp[head] = (0, head)

            else:
                tmp = [(0, rand_head) for rand_head in range(self.nentity)]
        elif self.mode == 'tail-batch':
            if(self.is_filter):
                # tmp = [(0, rand_tail) if (head, relation, rand_tail) not in self.triple_set
                #     else (-1, tail) for rand_tail in range(self.nentity)]
                # tmp[tail] = (0, tail)
                tmp = []
                for rand_tail in range(self.nentity):
                    if str(rand_tail) in self.ent_conc:
                        set_tail2conc = set(self.ent_conc[str(rand_tail)])
                    else:
                        set_tail2conc = set_rel2t
                    if (head, relation, rand_tail) not in self.triple_set:
                        if len(set_rel2t & set_tail2conc) > 0:
                            tmp.append((0, rand_tail))
                        else:
                            tmp.append((-1, tail))
                    else:
                        tmp.append((-1, tail))
                tmp[tail] = (0, tail)
            else:
                tmp = [(0, rand_tail) for rand_tail in range(self.nentity)]
        else:
            raise ValueError('negative batch mode %s not supported' % self.mode)

        tmp = torch.LongTensor(tmp)
        filter_bias = tmp[:, 0].float()
        negative_sample = tmp[:, 1]

        positive_sample = torch.LongTensor((head, relation, tail))

        return positive_sample, negative_sample, filter_bias, self.mode
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        filter_bias = torch.stack([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, filter_bias, mode
    
class BidirectionalOneShotIterator(object):
    def __init__(self, dataloader_head, dataloader_tail):
        self.iterator_head = self.one_shot_iterator(dataloader_head)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.step = 0
        
    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.iterator_head)
        else:
            data = next(self.iterator_tail)
        return data
    
    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data
