import torch.nn.functional as F

def BCEWithLogitLoss(negative_score, 
                     positive_score, 
                     subsampling_weight, 
                     negative_adversarial_sampling,
                     adversarial_temperature,
                     uni_weight,
                     model,
                     regularization):
    if negative_adversarial_sampling:
        #In self-adversarial sampling, we do not apply back-propagation on the sampling weight
        negative_score = (F.softmax(negative_score * adversarial_temperature, dim = 1).detach() 
                            * F.logsigmoid(-negative_score)).sum(dim = 1)
    else:
        negative_score = F.logsigmoid(-negative_score).mean(dim = 1)

    positive_score = F.logsigmoid(positive_score).squeeze(dim = 1)

    if uni_weight:
        positive_sample_loss = - positive_score.mean()
        negative_sample_loss = - negative_score.mean()
    else:
        positive_sample_loss = - (subsampling_weight * positive_score).sum()/subsampling_weight.sum()
        negative_sample_loss = - (subsampling_weight * negative_score).sum()/subsampling_weight.sum()

    loss = (positive_sample_loss + negative_sample_loss)/2
    
    if regularization != 0.0:
        #Use L3 regularization for ComplEx and DistMult
        regularization = regularization * (
            model.entity_embedding.norm(p = 3)**3 + 
            model.relation_embedding.norm(p = 3).norm(p = 3)**3
        )
        loss = loss + regularization
    else:
        regularization = None
    return loss, positive_sample_loss, negative_sample_loss, regularization


