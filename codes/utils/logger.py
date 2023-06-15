import logging
import os

def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''

    if args.do_train:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'train.log')
    else:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def show_test_progress(step, total_steps, test_log_steps):
    if step % test_log_steps == 0:
        logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

def test_step_log(logs, ranking):
    logs.append({
        'MRR': 1.0/ranking,
        'MR': float(ranking),
        'HITS@1': 1.0 if ranking <= 1 else 0.0,
        'HITS@3': 1.0 if ranking <= 3 else 0.0,
        'HITS@10': 1.0 if ranking <= 10 else 0.0,
    })
    return logs

def log_metrics(mode, step, logs):
    '''
    Print the evaluation logs
    '''
    metrics = {}
    for metric in logs[0].keys():
        metrics[metric] = sum([log[metric] for log in logs])/len(logs)
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))


def get_train_step_log(loss_name, loss_list, regularization):
    if regularization != None:
        regularization_log = {'regularization': regularization.item()}
    else:
        regularization_log = {}

    log = {
        **regularization_log,
    }
    for i in range(len(loss_list)):
        log[loss_name[i]] = loss_list[i]
    return log



def log_init_info( mode,
                   model_name,
                   init_step,
                   batch_size,
                   negative_sample_size,
                   hidden_dim,
                   gamma,
                   current_learning_rate,
                   is_filter,
                   uni_weight,
                   negative_adversarial_sampling,
                   adversarial_temperature = None):

    logging.info('Start %s' % mode)
    logging.info('Model: %s' % model_name)
    logging.info('init_step = %d' % init_step)
    logging.info('batch_size = %d' % batch_size)
    logging.info('negative_sample_size = %d' % negative_sample_size)
    logging.info('hidden_dim = %d' % hidden_dim)
    logging.info('gamma = %f' % gamma)
    logging.info('learning_rate = %f' % current_learning_rate)
    logging.info("is_filter = %s" % str(is_filter))
    logging.info("uni_weight = %s" % str(uni_weight))
    logging.info('negative_adversarial_sampling = %s' % str(negative_adversarial_sampling))
    if negative_adversarial_sampling:
        logging.info('adversarial_temperature = %f' % adversarial_temperature)
