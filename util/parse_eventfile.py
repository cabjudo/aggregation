import os.path

import numpy as np

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from glob import glob


experiments = glob('/NAS/home/graphs/*-0*')
print(experiments)

for exper in experiments:
    exper_name = exper.split('/')[-1]
    eventfile_path = '/NAS/home/graphs/' + exper_name + '/'

    savepath = '/NAS/archive/' + exper_name + '-training_metrics'
    if os.path.isfile(savepath + '.npz'):
        print(savepath, ' exists...')
        continue

    print('loading eventfiles...')
    event_acc = EventAccumulator(eventfile_path)
    event_acc.Reload()
    
    # get training metrics
    print('converting to numpy arrays...')
    loss = np.array(event_acc.Scalars('log/loss'))
    accuracy = np.array(event_acc.Scalars('log/accuracy'))
    
    # the files ran for 5 epochs on 49000 training samples
    # lets keep n samples per epoch
    n = 10.
    num_samples = len(loss)
    num_epochs = 5.
    samples_per_epoch = num_samples / num_epochs
    
    step_size = np.int(samples_per_epoch / n) # integer required for indexing
    
    # subsample training metrics
    print('subsampling data...')
    sampled_loss = loss[::step_size]
    sampled_accuracy = accuracy[::step_size]
    
    # save sampled loss
    print('saving to ', savepath, '...')
    np.savez_compressed(savepath, loss=sampled_loss, accuracy=sampled_accuracy)




