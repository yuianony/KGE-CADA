import torch
from model.loss import BCEWithLogitLoss
from tester.tester import Tester
import logging
import utils.checkpoint
import utils.logger
import torch.nn.functional as F

class Trainer():
    """
    Trainer class
    """
    def __init__(self, model, 
                       optimizer, 
                       train_iterator,
                       train_uniform_iterator,
                       init_step, 
                       warm_up_steps, 
                       current_learning_rate, 
                       all_true_triples, 
                       max_steps, 
                       save_checkpoint_steps, 
                       log_steps,
                       cuda, 
                       negative_adversarial_sampling,
                       adversarial_temperature,
                       uni_weight,
                       regularization,
                       args,
                       valid_steps, 
                       do_valid = False,
                       valid_dataset_list = None):

        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.train_iterator = train_iterator
        self.train_uniform_iterator = train_uniform_iterator
        self.init_step = init_step
        self.warm_up_steps = warm_up_steps
        self.current_learning_rate = current_learning_rate
        self.all_true_triples = all_true_triples
        self.max_steps = max_steps
        self.save_checkpoint_steps = save_checkpoint_steps
        self.log_steps = log_steps
        self.cuda = cuda
        self.negative_adversarial_sampling = negative_adversarial_sampling
        self.adversarial_temperature = adversarial_temperature
        self.uni_weight = uni_weight
        self.regularization = regularization
        self.args = args
        self.valid_steps = valid_steps
        self.do_valid = do_valid
        self.valid_dataset_list= valid_dataset_list
        self.training_logs = []
        self.updateP_step = 266


    def train_step(self):
        '''
        A single train step. Apply back-propation and return the loss
        '''
        positive_sample, negative_sample, subsampling_weight, mode, h_concept, t_concept = self.process_train_iterator(self.train_iterator)
        positive_sample_uniform, negative_sample_uniform, subsampling_weight_uniform, \
        mode_uniform, h_concept_uniform, t_concept_unifrom = self.process_train_iterator(self.train_uniform_iterator)

        self.optimizer.zero_grad()
        negative_score = self.model((positive_sample, negative_sample), mode=mode)
        positive_score = self.model(positive_sample)

        negative_score_uniform = self.model((positive_sample_uniform, negative_sample_uniform), mode=mode)
        positive_score_uniform = self.model(positive_sample_uniform)

        loss, positive_sample_loss, negative_sample_loss, regularization = \
        BCEWithLogitLoss( negative_score, 
                          positive_score, 
                          subsampling_weight, 
                          self.negative_adversarial_sampling,
                          self.adversarial_temperature,
                          self.uni_weight,
                          self.model,
                          self.regularization)

        uniform_loss, uniform_positive_sample_loss, uniform_negative_sample_loss, uniform_regularization = \
            BCEWithLogitLoss(negative_score_uniform,
                             positive_score_uniform,
                             subsampling_weight_uniform,
                             self.negative_adversarial_sampling,
                             self.adversarial_temperature,
                             self.uni_weight,
                             self.model,
                             0.0)

        score_c1, score_c2, ot_loss_h, ot_loss_t = self.model((positive_sample, h_concept, t_concept,
                                                          positive_sample_uniform, h_concept_uniform,
                                                          t_concept_unifrom), mode='concept-batch')

        score_c1 = F.logsigmoid(score_c1).squeeze(dim=1)
        score_c2 = F.logsigmoid(score_c2).squeeze(dim=1)
        if self.uni_weight:
            c1_loss = -score_c1.mean()
            c2_loss = -score_c2.mean()
        else:
            c1_loss = - (subsampling_weight * score_c1).sum() / subsampling_weight.sum()
            c2_loss = - (subsampling_weight_uniform * score_c2).sum() / subsampling_weight.sum()

        loss += uniform_loss + self.args.a1 * (c1_loss + c2_loss) + self.args.a2*(ot_loss_h + ot_loss_t + self.model.concept_embedding.norm(p = 3)**3)

        loss.backward()
        self.optimizer.step()

        loss_list = [loss, positive_sample_loss, negative_sample_loss, uniform_loss, c1_loss, c2_loss, ot_loss_h, ot_loss_t]
        loss_name = ["loss", "positive_sample_loss", "negative_sample_loss", "uniform_loss", "c1_loss", "c2_loss", "ot_loss_h", "ot_loss_t"]

        log = utils.logger.get_train_step_log(loss_name, loss_list, regularization)
        return log
    
    def train_loop(self):

        self.model.train()
        step = self.init_step
        for step in range(self.init_step, self.max_steps):

            log = self.train_step()
            
            self.training_logs.append(log)
            if step % self.log_steps ==0:
                utils.logger.log_metrics("Training average", step, self.training_logs)
                self.training_logs = []

            self.warm_up(step)
            #if step > 0 and step % self.updateP_step == 0:
            #    self.model.update_P()

            if step % self.save_checkpoint_steps == 0:
                utils.checkpoint.save_model(self.model, 
                                      self.optimizer, 
                                      step, 
                                      self.current_learning_rate, 
                                      self.warm_up_steps, 
                                      self.args)

            self.valid(step)

        utils.checkpoint.save_model(self.model, 
                                self.optimizer, 
                                step, 
                                self.current_learning_rate, 
                                self.warm_up_steps, 
                                self.args)
        return step

    def process_train_iterator(self, train_iterator):
        positive_sample, negative_sample, subsampling_weight, mode, h_concept, t_concept = next(train_iterator)
        if self.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()
            h_concept = h_concept.cuda()
            t_concept = t_concept.cuda()
        return positive_sample, negative_sample, subsampling_weight, mode, h_concept, t_concept


    def warm_up(self, step):
        if step >= self.warm_up_steps:
            self.current_learning_rate = self.current_learning_rate / 10
            logging.info('Change learning_rate to %f at step %d' % (self.current_learning_rate, step))
            self.optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()), 
                lr=self.current_learning_rate
            )
            self.warm_up_steps = self.warm_up_steps * 3


    def valid(self, step):
        if self.do_valid and step % self.valid_steps == 0:
            logging.info('Evaluating on Valid Dataset...')
            valid_object = Tester('Valid',self.model,self.valid_dataset_list,1000,step,self.cuda,self.args)
            valid_object.test_step()
