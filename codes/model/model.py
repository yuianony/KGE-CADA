#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
from gw_ot.wgw import Entropic_WGW
class KGEModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, nconcept, hidden_dim, gamma, ent_dom, head_ent, tail_ent,
                 double_entity_embedding=False, double_relation_embedding=False):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0

        self.wgw = Entropic_WGW()
        self.P = torch.ones((self.nentity, self.nentity), dtype=torch.float)
        #self.initP(ent_dom, head_ent, tail_ent)
        self.alpha = 1.0
        
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]), 
            requires_grad=False
        )
        
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]), 
            requires_grad=False
        )
        
        self.entity_dim = hidden_dim*2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim*2 if double_relation_embedding else hidden_dim


        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )
        
        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )

        self.concept_embedding = nn.Parameter(torch.zeros(nconcept, self.entity_dim))
        nn.init.uniform_(
            tensor=self.concept_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.w = nn.Parameter(torch.ones(1, nconcept))
        nn.init.uniform_(
            tensor=self.w,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.A1 = torch.zeros(nentity, self.entity_dim)  # store the concept emb for every entity in trainset
        self.A2 = torch.zeros(nentity, self.entity_dim)  # store the concept emb for every entity in trainset_uniform

        if model_name == 'pRotatE':
            self.modulus = nn.Parameter(torch.Tensor([[0.5 * self.embedding_range.item()]]))
        
        #Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'pRotatE']:
            raise ValueError('model %s not supported' % model_name)
            
        if model_name == 'RotatE' and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('RotatE should use --double_entity_embedding')

        if model_name == 'ComplEx' and (not double_entity_embedding or not double_relation_embedding):
            raise ValueError('ComplEx should use --double_entity_embedding and --double_relation_embedding')
        
    def forward(self, sample, mode='single'):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements 
        in their triple ((head, relation) or (relation, tail)).
        '''

        if mode == 'single':
            batch_size, negative_sample_size = sample.size(0), 1
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=sample[:,0]
            ).unsqueeze(1)
            
            relation = torch.index_select(
                self.relation_embedding, 
                dim=0, 
                index=sample[:,1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=sample[:,2]
            ).unsqueeze(1)
            
        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            
            relation = torch.index_select(
                self.relation_embedding, 
                dim=0, 
                index=tail_part[:, 1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=tail_part[:, 2]
            ).unsqueeze(1)
            
        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=head_part[:, 0]
            ).unsqueeze(1)
            
            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

        elif mode == 'head-batch-concept':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

            head = torch.index_select(
                self.A1,
                dim=0,
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.A1,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)

        elif mode == 'tail-batch-concept':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            head = torch.index_select(
                self.A1,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.A1,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

        elif mode == 'concept-batch':
            positive_sample, h1_concept, t1_concept, positive_sample_uniform, h2_concept, t2_concept = sample

            h1_idx = positive_sample[:, 0]
            t1_idx = positive_sample[:, 2]
            h2_idx = positive_sample_uniform[:, 0]
            t2_idx = positive_sample_uniform[:, 2]

            h1 = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=positive_sample[:, 0]
            )

            r1 = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=positive_sample[:, 1]
            )

            t1 = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=positive_sample[:, 2]
            )

            h2 = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=positive_sample_uniform[:, 0]
            )

            r2 = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=positive_sample_uniform[:, 1]
            )

            t2 = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=positive_sample_uniform[:, 2]
            )

            h1_c = torch.mm(h1_concept * self.w, self.concept_embedding)
            t1_c = torch.mm(t1_concept * self.w, self.concept_embedding)
            h2_c = torch.mm(h2_concept * self.w, self.concept_embedding)
            t2_c = torch.mm(t2_concept * self.w, self.concept_embedding)

            h1_c = torch.mul(h1, h1_c)
            t1_c = torch.mul(t1, t1_c)
            h2_c = torch.mul(h2, h2_c)
            t2_c = torch.mul(t2, t2_c)

            self.A1[h1_idx] = h1_c
            self.A1[t1_idx] = t1_c
            self.A2[h2_idx] = h2_c
            self.A2[t2_idx] = t2_c
            
        else:
            raise ValueError('mode %s not supported' % mode)
            
        model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'pRotatE': self.pRotatE
        }
        
        if self.model_name in model_func:
            if mode == 'concept-batch':
                score_1 = model_func[self.model_name](h1_c.unsqueeze(1), r1.unsqueeze(1), t1_c.unsqueeze(1), mode)
                score_2 = model_func[self.model_name](h2_c.unsqueeze(1), r2.unsqueeze(1), t2_c.unsqueeze(1), mode)

                ot_loss_h = 0.0
                ot_loss_t = 0.0

                ot_loss_h = self.wgw_loss(h1_idx, h2_idx, h1_c, h2_c)
                ot_loss_t = self.wgw_loss(t1_idx, t2_idx, t1_c, t2_c)
                return score_1, score_2, ot_loss_h, ot_loss_t

            else:
                score = model_func[self.model_name](head, relation, tail, mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)
        
        return score
    
    def TransE(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def DistMult(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim = 2)
        return score

    def ComplEx(self, head, relation, tail, mode):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim = 2)
        return score

    def RotatE(self, head, relation, tail, mode):
        pi = 3.14159265358979323846
        
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        #Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation/(self.embedding_range.item()/pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0)

        score = self.gamma.item() - score.sum(dim = 2)
        return score

    def pRotatE(self, head, relation, tail, mode):
        pi = 3.14159262358979323846
        
        #Make phases of entities and relations uniformly distributed in [-pi, pi]

        phase_head = head/(self.embedding_range.item()/pi)
        phase_relation = relation/(self.embedding_range.item()/pi)
        phase_tail = tail/(self.embedding_range.item()/pi)

        if mode == 'head-batch':
            score = phase_head + (phase_relation - phase_tail)
        else:
            score = (phase_head + phase_relation) - phase_tail

        score = torch.sin(score)            
        score = torch.abs(score)

        score = self.gamma.item() - score.sum(dim = 2) * self.modulus
        return score

    def wgw_loss(self, ent_1, ent_2, emb1, emb2):

        P_sliced = self.P[ent_1][:, ent_2]

        # compute w_cost
        if self.wgw.lamda == 0.:
            w_cost = torch.tensor(0.)
        else:
            norm = self.wgw.cost_matrix(emb1, emb2, cost_type='L2')
            w_cost = torch.sum(norm * P_sliced)

        # compute gw_cost
        if self.wgw.lamda == 1.:
            gw_cost = torch.tensor(0.)
        else:
            C1 = self.wgw.cost_matrix(emb1, emb1, cost_type=self.wgw.intra_cost_type)
            C2 = self.wgw.cost_matrix(emb2, emb2, cost_type=self.wgw.intra_cost_type)

            L = self.wgw.tensor_matrix_mul(C1, C2, P_sliced)

            gw_cost = (P_sliced * L).sum()


        return self.wgw.lamda * w_cost + (1 - self.wgw.lamda) * gw_cost

    def initP(self, ent_dom, head_ent, tail_ent):
        for h in head_ent:
            for t in tail_ent:
                if str(h) not in ent_dom or str(t) not in ent_dom:
                    continue
                else:
                    a = ent_dom[str(h)]
                    b = ent_dom[str(t)]
                    self.P[h][t] = len(set(a).intersection(set(b)))/len(set(a).union(set(b)))
        return
    def update_P(self):
        # update transport plan P after some epochs of training
        # n1: 0 ~ (n1-1) entities belong to the first domain
        # n2: n2 ~ : entities belong to the second domain

        if self.alpha == 0.0:
            return torch.tensor(0.0)

        with torch.no_grad():
            skn_cost, tmp_P = self.wgw(self.A1, self.A2)
            self.P = tmp_P

            if self.wgw.verbose:
                print('tmp_P: ', tmp_P.sum(), '\n', tmp_P)
                print('tmp_P: ', tmp_P.sum(dim=0), tmp_P.sum(dim=1))

        return skn_cost
