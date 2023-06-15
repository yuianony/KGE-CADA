import torch
import logging
import os
import json
import numpy as np

def save_variable(step, current_learning_rate, warm_up_steps):
    save_variable_list = { 
        'step': step, 
        'current_learning_rate': current_learning_rate,
        'warm_up_steps': warm_up_steps
    }
    return save_variable_list

def save_model(model, optimizer, step, current_learning_rate, warm_up_steps, args):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''
    
    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    save_variable_list = save_variable(step, current_learning_rate, warm_up_steps)
    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(args.save_path, 'checkpoint')
    )
    
    entity_embedding = model.entity_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'entity_embedding'), 
        entity_embedding
    )
    
    relation_embedding = model.relation_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'relation_embedding'), 
        relation_embedding
    )

def read_checkpoint(model, optimizer, init_checkpoint, do_train, learning_rate, warm_up_steps):
    # Restore model from checkpoint directory
    current_learning_rate = learning_rate
    warm_up_steps = warm_up_steps
    logging.info('Loading checkpoint %s...' % init_checkpoint)
    checkpoint = torch.load(os.path.join(init_checkpoint, 'checkpoint'))
    init_step = checkpoint['step']
    model.load_state_dict(checkpoint['model_state_dict'])
    if do_train:
        current_learning_rate = checkpoint['current_learning_rate']
        warm_up_steps = checkpoint['warm_up_steps']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return init_step, current_learning_rate, warm_up_steps

def override_config(args):
    '''
    Override model and data configuration
    '''
    
    with open(os.path.join(args.init_checkpoint, 'config.json'), 'r') as fjson:
        argparse_dict = json.load(fjson)
    
    if args.data_path is None:
        args.data_path = argparse_dict['data_path']
    args.model = argparse_dict['model']
    args.double_entity_embedding = argparse_dict['double_entity_embedding']
    args.double_relation_embedding = argparse_dict['double_relation_embedding']
    args.hidden_dim = argparse_dict['hidden_dim']
    args.test_batch_size = argparse_dict['test_batch_size']
