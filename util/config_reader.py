import configparser
import argparse

config = configparser.ConfigParser()
config.read_dict({'dataset_options': {'datadir': '/home/christine/projects/convnet/datasets/cifar-10-batches-py',
                                      'data_type': 'CIFAR10'},
                  'dataset_params': {'train_size': 49000,
                                     'test_size': 1000,
                                     'val_size': 1000},
                  'model_options': {'network': 'relu'},
                  'model_params': {'indim': '32, 32, 3',
                                   'strides': '1, 1, 2, 1, 1, 2, 1',
                                   'padding': 'valid, valid, valid, valid, valid, valid, valid',
                                   'kernel_size': '3, 3, 3, 3, 3, 3, 4',
                                   'filters': '16, 16, 16, 16, 16, 16, 10'},
                  'logging_options': {'state_freq': 1,
                                      'metric_freq': 5},
                  'logging_params': {'savedir': '/home/christine/projects/convnet/paper/checkpoints/',
                                     'logdir': '/home/christine/projects/convnet/paper/graphs/'},
                  'training_options': {'lr': 0.01,
                                       'batch_size': 64,
                                       'num_epochs': 5},
                  'training_params': {'allow_soft_placement': True,
                                      'print_freq': 100,
                                      'log_device_placement': True}})



def get_dataset_options(configfile):
    config.read(configfile)
    
    dataset_options = dict(config.items('dataset_options'))

    return dataset_options


def get_dataset_params(configfile):
    config.read(configfile)
    
    dataset_params = dict(config.items('dataset_params'))
    # map all values to ints
    for key, val in dataset_params.items():
        dataset_params[key] = int(val)

    return dataset_params


def get_model_options(configfile):
    config.read(configfile)
    
    model_options = dict(config.items('model_options'))

    return model_options


def get_model_params(configfile):
    config.read(configfile)
    
    model_params = dict(config.items('model_params'))
    # map values to int list
    for k,v in model_params.items():
        if k == 'padding':
            model_params[k] = model_params[k].split(', ')
            continue

        model_params[k] = list(map(int, model_params[k].split(', ')))

    return model_params

    
def get_logging_options(configfile):
    config.read(configfile)

    logging_options = dict(config.items('logging_options'))
    # map all values to ints
    for key, val in logging_options.items():
        logging_options[key] = int(val)

    return logging_options
    

def get_logging_params(configfile):
    config.read(configfile)

    logging_params = dict(config.items('logging_params'))

    return logging_params


def get_training_options(configfile):
    config.read(configfile)

    training_options = dict(config.items('training_options'))
    # map all values to ints
    for key, val in training_options.items():
        if key == 'lr':
            training_options[key] = float(val)
            continue
        
        training_options[key] = int(val)

    return training_options


def get_training_params(configfile):
    config.read(configfile)

    training_params = dict(config.items('training_params'))
    # map values to int list
    for k,v in training_params.items():
        if k == 'print_freq':
            training_params[k] = int(training_params[k])
            continue

        training_params[k] = config.getboolean('training_params', k)
    
    return training_params


def get_dataset(configfile):
    dataset_options = get_dataset_options(configfile)
    dataset_params = get_dataset_params(configfile)

    return {'options': dataset_options, 'params': dataset_params}


def get_model(configfile):
    model_options = get_model_options(configfile)
    model_params = get_model_params(configfile)

    return {'options': model_options, 'params': model_params}


def get_logging(configfile):
    logging_options = get_logging_options(configfile)
    logging_params = get_logging_params(configfile)

    return {'options': logging_options, 'params': logging_params}


def get_training(configfile):
    training_options = get_training_options(configfile)
    training_params = get_training_params(configfile)

    return {'options': training_options, 'params': training_params}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parse config file')
    parser.add_argument('-configfile', default='../config/default.ini')
    args = parser.parse_args()

    configfile = args.configfile
    # dataset
    dataset_options = get_dataset_options(configfile)
    dataset_params = get_dataset_params(configfile)
    print('dataset', dataset_options, dataset_params)

    # model
    model_options = get_model_options(configfile)
    model_params = get_model_params(configfile)
    print('model', model_options, model_params)
    
    # logging
    logging_options = get_logging_options(configfile)
    logging_params = get_logging_params(configfile)
    print('logging', logging_options, logging_params)
    
    # training
    training_options = get_training_options(configfile)
    training_params = get_training_params(configfile)
    print('training', training_options, training_params)
