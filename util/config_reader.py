import configparser


def get_dataset_options(configfile):
    config = configparser.ConfigParser()
    config.read(configfile)
    
    dataset_options = dict(config.items('dataset_options'))

    return dataset_options


def get_dataset_params(configfile):
    config = configparser.ConfigParser()
    config.read(configfile)
    
    dataset_params = dict(config.items('dataset_params'))
    # map all values to ints
    for key, val in dataset_params.items():
        dataset_params[key] = int(val)

    return dataset_params


def get_model_options(configfile):
    config = configparser.ConfigParser()
    config.read(configfile)
    
    model_options = dict(config.items('model_options'))

    return model_options


def get_model_params(configfile):
    config = configparser.ConfigParser()
    config.read(configfile)
    
    model_params = dict(config.items('model_params'))
    # map values to int list
    for k,v in model_params.items():
        if k == 'padding':
            model_params[k] = model_params[k].split(', ')
            continue

        model_params[k] = map(int, model_params[k].split(','))

    return model_params

    
def get_logging_options(configfile):
    config = configparser.ConfigParser()
    config.read(configfile)

    logging_options = dict(config.items('logging_options'))
    # map all values to ints
    for key, val in logging_options.items():
        logging_options[key] = int(val)

    return logging_options
    

def get_logging_params(configfile):
    config = configparser.ConfigParser()
    config.read(configfile)

    logging_params = dict(config.items('logging_params'))

    return logging_params


def get_training_options(configfile):
    config = configparser.ConfigParser()
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
    config = configparser.ConfigParser()
    config.read(configfile)

    training_params = dict(config.items('training_params'))
    # map values to int list
    for k,v in training_params.items():
        if k == 'print_freq':
            training_params[k] = int(training_params[k])
            continue

        training_params[k] = config.getboolean('training_params', k)
    
    return training_params



if __name__ == '__main__':
    configfile = '../config/default.ini'
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



















