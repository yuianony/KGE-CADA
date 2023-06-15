import os 
import logging
import json

def read_triple(file_path, entity2id, relation2id):
    '''
    Read triples and map them into ids.
    '''
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples

def get_entid(file_path, entity2id):
    '''
    Get entity id
    '''
    entid = []
    with open(file_path) as fin:
        for line in fin:
            ent_name = line.strip('\n')
            entid.append(entity2id[ent_name])
    return entid

def get_entity2id(data_path):
    '''
    Get the mapping of entity to id
    '''
    with open(os.path.join(data_path, 'entities.dict')) as fin:
        entity2id = dict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)
    return entity2id

def get_relation2id(data_path):
    '''
    Get the mapping of relation to id
    '''
    with open(os.path.join(data_path, 'relations.dict')) as fin:
        relation2id = dict()
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)
    return relation2id

def read_data_from_datapath(data_path,sp):
    '''
    get the num of entity, relation and triples
    return the num of entity and relation , all triples
    '''
    entity2id = get_entity2id(data_path)
    relation2id = get_relation2id(data_path)
    nentity = len(entity2id)
    nrelation = len(relation2id)

    with open(os.path.join(data_path, 'rel2dom_h.json')) as fin:
        rel2dom_h = json.load(fin)
    with open(os.path.join(data_path, 'rel2dom_t.json')) as fin:
        rel2dom_t = json.load(fin)

    with open(os.path.join(data_path, 'dom_ent.json')) as fin:
        dom_ent = json.load(fin)
    with open(os.path.join(data_path, 'ent_dom.json')) as fin:
        ent_dom = json.load(fin)

    with open(os.path.join(data_path, 'rel2nn.json')) as fin:
        rel2nn = json.load(fin)

    nconcept = len(dom_ent)
    
    logging.info('Data Path: %s' % data_path)
    logging.info('#entity: %d' % nentity)
    logging.info('#relation: %d' % nrelation)

    sp_threshold = sp

    train_triples = read_triple(os.path.join(data_path, 'train.txt'), entity2id, relation2id)
    logging.info('#train: %d' % len(train_triples))

    all_valid_triples = read_triple(os.path.join(data_path, 'valid.txt'), entity2id, relation2id)
    unifrom_valid_triples_path = 'validData/sp_' + str(sp_threshold) + '/valid_split_1.txt'
    valid_triples = read_triple(os.path.join(data_path, unifrom_valid_triples_path), entity2id, relation2id)
    logging.info('#valid: %d' % len(valid_triples))

    all_test_triples = read_triple(os.path.join(data_path, 'test.txt'), entity2id, relation2id)
    unifrom_test_triples_path = 'testData/sp_' + str(sp_threshold) + '/test_split_1.txt'
    test_triples = read_triple(os.path.join(data_path, unifrom_test_triples_path), entity2id, relation2id)
    logging.info('#test: %d' % len(test_triples))

    unifrom_train_triples_path = 'trainData/sp_' + str(sp_threshold) + '/train_split_1.txt'
    train_triples_uniform = read_triple(os.path.join(data_path, unifrom_train_triples_path), entity2id,
                                        relation2id)

    #All true triples
    all_true_triples = train_triples + all_valid_triples + all_test_triples

    head_ent_path = 'trainData/sp_' + str(sp_threshold) + '/ent_0.txt'
    tail_ent_path = 'trainData/sp_' + str(sp_threshold) + '/ent_1.txt'
    head_ent = get_entid(os.path.join(data_path, head_ent_path), entity2id)
    tail_ent = get_entid(os.path.join(data_path, tail_ent_path), entity2id)

    return nentity, nrelation, all_true_triples, train_triples, valid_triples, test_triples, \
           train_triples_uniform, ent_dom, nconcept, rel2dom_h, rel2dom_t, head_ent, tail_ent
