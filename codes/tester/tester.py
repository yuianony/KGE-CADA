import utils.logger
import torch
class Tester():
    def __init__(self, mode, model, test_dataset_list, test_log_steps, model_steps, cuda, args) -> None:
        self.mode = mode
        self.model = model
        self.test_dataset_list = test_dataset_list
        self.test_log_steps = test_log_steps
        self.cuda = cuda
        self.model_steps = model_steps
        self.logs = []
        self.args = args

    ### has bug ###
    #def process_data(self):
    #    for test_dataset in self.test_dataset_list:
    #        for positive_sample, negative_sample, filter_bias, _ in test_dataset:
    #            if self.cuda:
    #                positive_sample = positive_sample.cuda()
    #                negative_sample = negative_sample.cuda()
    #                filter_bias = filter_bias.cuda()
    

    def test_step(self):
        '''
        Evaluate the model on test or valid datasets
        '''
        
        self.model.eval()
        #self.process_data()

        step = 0
        total_steps = sum([len(dataset) for dataset in self.test_dataset_list])

        with torch.no_grad():
            for test_dataset in self.test_dataset_list:
                for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                    if self.cuda:
                        positive_sample = positive_sample.cuda()
                        negative_sample = negative_sample.cuda()
                        filter_bias = filter_bias.cuda()
                    batch_size = positive_sample.size(0)

                    score = self.model((positive_sample, negative_sample), mode)
                    if mode == 'head-batch':
                        score_c = self.model((positive_sample, negative_sample), mode='head-batch-concept')
                    elif mode == 'tail-batch':
                        score_c = self.model((positive_sample, negative_sample), mode='tail-batch-concept')
                    score += score_c * self.args.a1
                    score += filter_bias

                    #Explicitly sort all the entities to ensure that there is no test exposure bias
                    argsort = torch.argsort(score, dim = 1, descending=True)

                    if mode == 'head-batch':
                        positive_arg = positive_sample[:, 0]
                    elif mode == 'tail-batch':
                        positive_arg = positive_sample[:, 2]
                    else:
                        raise ValueError('mode %s not supported' % mode)

                    for i in range(batch_size):
                        #Notice that argsort is not ranking
                        ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                        assert ranking.size(0) == 1

                        #ranking + 1 is the true ranking used in evaluation metrics
                        ranking = 1 + ranking.item()

                        self.logs = utils.logger.test_step_log(self.logs, ranking)

                    utils.logger.show_test_progress(step,total_steps,self.test_log_steps)
                    step += 1

        utils.logger.log_metrics(self.mode ,self.model_steps,self.logs)
