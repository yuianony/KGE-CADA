#!/usr/bin/python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import logging
import torch
import utils.logger 
import utils.checkpoint
from trainer.trainer import Trainer
from tester.tester import Tester
from dataloader.dataprocess import  read_data_from_datapath
from utils.args import parse_args

from torch.utils.data import DataLoader

from model import KGEModel

from dataloader import TrainDataset,TestDataset
from dataloader import BidirectionalOneShotIterator
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


        
def main(args):
    if args.init_checkpoint:
        utils.checkpoint.override_config(args)

    do_train = args.do_train
    do_valid = args.do_valid
    do_test = args.do_test
    init_checkpoint = args.init_checkpoint
    data_path = args.data_path
    save_path = args.save_path
    model_name = args.model
    hidden_dim = args.hidden_dim
    gamma = args.gamma
    double_entity_embedding = args.double_entity_embedding
    double_relation_embedding = args.double_relation_embedding
    cuda = args.cuda
    negative_sample_size = args.negative_sample_size
    train_batch_size = args.train_batch_size
    cpu_num = args.cpu_num
    learning_rate = args.learning_rate
    max_steps = args.max_steps
    evaluate_train = args.evaluate_train
    cuda = args.cuda
    negative_adversarial_sampling = args.negative_adversarial_sampling
    adversarial_temperature = args.adversarial_temperature
    uni_weight = args.uni_weight
    regularization = args.regularization
    valid_steps = args.valid_steps
    save_checkpoint_steps = args.save_checkpoint_steps
    warm_up_steps = args.warm_up_steps
    test_batch_size = args.test_batch_size
    is_filter = args.filter
    test_log_steps = args.test_log_steps
    log_steps = args.log_steps
    sp = args.sp

    if (not do_train) and (not do_valid) and (not do_test):
        raise ValueError('one of train/val/test mode must be choosed.')
    

    elif data_path is None:
        raise ValueError('one of init_checkpoint/data_path must be choosed.')

    if do_train and save_path is None:
        raise ValueError('Where do you want to save your trained model?')
    
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Write logs to checkpoint and console
    utils.logger.set_logger(args)
    
    #entity2id = get_entity2id(data_path)
    #relation2id = get_relation2id(data_path)
    #nentity = len(entity2id)
    #nrelation = len(relation2id)
    #
    #args.nentity = nentity
    #args.nrelation = nrelation
    #
    #logging.info('Data Path: %s' % data_path)
    #logging.info('#entity: %d' % nentity)
    #logging.info('#relation: %d' % nrelation)
    #
    #train_triples = read_triple(os.path.join(data_path, 'train.txt'), entity2id, relation2id)
    #logging.info('#train: %d' % len(train_triples))
    #valid_triples = read_triple(os.path.join(data_path, 'valid.txt'), entity2id, relation2id)
    #logging.info('#valid: %d' % len(valid_triples))
    #test_triples = read_triple(os.path.join(data_path, 'test.txt'), entity2id, relation2id)
    #logging.info('#test: %d' % len(test_triples))
    #
    ##All true triples
    #all_true_triples = train_triples + valid_triples + test_triples
    nentity, nrelation, all_true_triples, train_triples, valid_triples, test_triples, \
    train_triples_uniform, ent_dom, nconcept, rel2dom_h, rel2dom_t, head_ent, tail_ent = read_data_from_datapath(data_path,sp)
    args.nentity = nentity
    args.nrelation = nrelation

    
    kge_model = KGEModel(
        model_name=model_name,
        nentity=nentity,
        nrelation=nrelation,
        nconcept=nconcept,
        hidden_dim=hidden_dim,
        gamma=gamma,
        ent_dom=ent_dom,
        head_ent=head_ent,
        tail_ent=tail_ent,
        double_entity_embedding=double_entity_embedding,
        double_relation_embedding=double_relation_embedding
    )
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, kge_model.parameters()), 
        lr=learning_rate
    )
    step = 0
    current_learning_rate = learning_rate

    if cuda:
        kge_model = kge_model.cuda()
        kge_model.A1 = kge_model.A1.cuda()
        kge_model.A2 = kge_model.A2.cuda()
        kge_model.P = kge_model.P.cuda()
    
    valid_dataloader_head = DataLoader(
        TestDataset(
            valid_triples, 
            all_true_triples, 
            nentity, 
            nrelation, 
            'head-batch',
            rel2dom_h,
            rel2dom_t,
            ent_dom,
            is_filter
        ), 
        batch_size=test_batch_size,
        num_workers=max(1, cpu_num//2), 
        collate_fn=TestDataset.collate_fn
    )

    valid_dataloader_tail = DataLoader(
        TestDataset(
            valid_triples, 
            all_true_triples, 
            nentity, 
            nrelation, 
            'tail-batch',
            rel2dom_h,
            rel2dom_t,
            ent_dom,
            is_filter
        ), 
        batch_size=test_batch_size,
        num_workers=max(1, cpu_num//2), 
        collate_fn=TestDataset.collate_fn
    )
    valid_dataset_list = [valid_dataloader_head, valid_dataloader_tail]
    if do_train:
        # Set training dataloader iterator
        train_dataloader_head = DataLoader(
            TrainDataset(train_triples, nentity, nrelation, negative_sample_size, 'head-batch', ent_dom, nconcept, rel2dom_h, rel2dom_t),
            batch_size=train_batch_size,
            shuffle=True, 
            num_workers=max(1, cpu_num//2),
            collate_fn=TrainDataset.collate_fn
        )
        train_dataloader_head = list(train_dataloader_head)
        
        train_dataloader_tail = DataLoader(
            TrainDataset(train_triples, nentity, nrelation, negative_sample_size, 'tail-batch', ent_dom, nconcept, rel2dom_h, rel2dom_t),
            batch_size=train_batch_size,
            shuffle=True, 
            num_workers=max(1, cpu_num//2),
            collate_fn=TrainDataset.collate_fn
        )
        train_dataloader_tail = list(train_dataloader_tail)

        train_dataloader_head_uniform = DataLoader(
            TrainDataset(train_triples_uniform, nentity, nrelation, args.negative_sample_size, 'head-batch', ent_dom, nconcept, rel2dom_h, rel2dom_t),
            batch_size=train_batch_size // 2,
            shuffle=True,
            num_workers=max(1, cpu_num // 2),
            collate_fn=TrainDataset.collate_fn
        )
        train_dataloader_head_uniform = list(train_dataloader_head_uniform)

        train_dataloader_tail_uniform = DataLoader(
            TrainDataset(train_triples_uniform, nentity, nrelation, args.negative_sample_size, 'tail-batch', ent_dom, nconcept, rel2dom_h, rel2dom_t),
            batch_size=train_batch_size // 2,
            shuffle=True,
            num_workers=max(1, cpu_num // 2),
            collate_fn=TrainDataset.collate_fn
        )
        train_dataloader_tail_uniform = list(train_dataloader_tail_uniform)
        
        train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)
        train_iterator_uniform = BidirectionalOneShotIterator(train_dataloader_head_uniform,
                                                              train_dataloader_tail_uniform)

        
        # Set training configuration
        current_learning_rate = learning_rate
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, kge_model.parameters()), 
            lr=current_learning_rate
        )
        if warm_up_steps:
            warm_up_steps = warm_up_steps
        else:
            warm_up_steps = max_steps // 2

    
    # read hyper parameters from checkpoint 
    if init_checkpoint:
        init_step, current_learning_rate, warm_up_steps = utils.checkpoint.read_checkpoint(kge_model, optimizer, init_checkpoint,do_train, learning_rate, warm_up_steps)
    else:
        logging.info('Ramdomly Initializing %s Model...'% model_name)
        init_step = 0
    
    # Set valid dataloader as it would be evaluated during training
    
    if do_train:
        utils.logger.log_init_info( "train",
                                    model_name,
                                    init_step,
                                    train_batch_size,
                                    negative_sample_size,
                                    hidden_dim,
                                    gamma,
                                    current_learning_rate,
                                    is_filter,
                                    uni_weight,
                                    negative_adversarial_sampling,
                                    adversarial_temperature)
        trainer_object = Trainer(kge_model,
                                 optimizer,
                                 train_iterator,
                                 train_iterator_uniform,
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
                                 do_valid,
                                 valid_dataset_list)
        step = trainer_object.train_loop()
        
    if do_valid:
        utils.logger.log_init_info( "valid",
                                    model_name,
                                    init_step,
                                    test_batch_size,
                                    negative_sample_size,
                                    hidden_dim,
                                    gamma,
                                    current_learning_rate,
                                    is_filter,
                                    uni_weight,
                                    negative_adversarial_sampling,
                                    adversarial_temperature)
        valid_object = Tester('Valid',kge_model,valid_dataset_list,1000,step,cuda,args)
        valid_object.test_step()
    
    if do_test:
        utils.logger.log_init_info( "test",
                                    model_name,
                                    init_step,
                                    test_batch_size,
                                    negative_sample_size,
                                    hidden_dim,
                                    gamma,
                                    current_learning_rate,
                                    is_filter,
                                    uni_weight,
                                    negative_adversarial_sampling,
                                    adversarial_temperature)
        test_dataloader_head = DataLoader(
            TestDataset(
                test_triples, 
                all_true_triples, 
                nentity, 
                nrelation, 
                'head-batch',
                rel2dom_h,
                rel2dom_t,
                ent_dom,
                is_filter
            ), 
            batch_size=test_batch_size,
            num_workers=max(1, cpu_num//2), 
            collate_fn=TestDataset.collate_fn
        )

        test_dataloader_tail = DataLoader(
            TestDataset(
                test_triples, 
                all_true_triples, 
                nentity, 
                nrelation, 
                'tail-batch',
                rel2dom_h,
                rel2dom_t,
                ent_dom,
                is_filter
            ), 
            batch_size=test_batch_size,
            num_workers=max(1, cpu_num//2), 
            collate_fn=TestDataset.collate_fn
        )
        
        test_dataset_list = [test_dataloader_head, test_dataloader_tail]
        test_head_dataset_list = [test_dataloader_head]
        test_tail_dataset_list = [test_dataloader_tail]
        tester_object = Tester('Test',kge_model,test_dataset_list,test_log_steps,step,cuda,args)
        tester_object.test_step()
        tester_head_object = Tester('Test Head',kge_model,test_head_dataset_list,test_log_steps,step,cuda,args)
        tester_head_object.test_step()
        tester_tail_object = Tester('Test Tail',kge_model,test_tail_dataset_list,test_log_steps,step,cuda,args)
        tester_tail_object.test_step()
    #if evaluate_train:
    #    logging.info('Evaluating on Training Dataset...')
    #    metrics = kge_model.test_step(kge_model, train_triples, all_true_triples, args)
    #    log_metrics('Test', step, metrics)
        
if __name__ == '__main__':
    main(parse_args())
