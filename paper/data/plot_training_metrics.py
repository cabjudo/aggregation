import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

from glob import glob


def data_check(d):
    d_check = d[0,:].reshape(1, -1)
    for pt in d[1:]:
        if d_check[-1, 1] < pt[1]:
            d_check = np.vstack((d_check, pt))

    return d_check


def load_experiment_data(filenames):
    data = {}
    
    for exper in filenames:
        # name-training-metrics.npz
        name = exper.split('-t')[0]
        data[name] = dict(np.load(exper))
        data[name]['accuracy'] = data_check(data[name]['accuracy'])

    return data


def display_data(data):
    plt.figure(figsize=(100,100))
    
    handles = []
    prefix = ''
    for k,v in data.items():
        prefix = k.split('/')[-1]
        handles.append( plt.plot(data[k]['accuracy'][:, 1], data[k]['accuracy'][:, 2], label=prefix)[0] )

    plt.legend(handles=handles, framealpha=0.5, frameon=False, fontsize='small')
    
    plt.title( prefix + '-lr')
    plt.ylabel('accuracy')
    plt.xlabel('sample')
    plt.show(block=False)


def display_griddata(data, savepath=None):
    '''
    data is a list of data
    '''
    n = len(data)
    # The best fit grid
    height = np.int( np.sqrt(n) )
    width = np.int(np.ceil( n / height ))
    ratio = height/width

    fig_height = 20
    fig_width = fig_height/ratio
    plt.figure(figsize=(fig_height, fig_width))
    
    for h in np.arange(height):
        for w in np.arange(width):
            idx = w * height + h
            if (idx >= n):
                break
            # get the data for this cell
            d = data[idx]
            handles = []
            # select the subplot
            plt.subplot(width, height, idx + 1)
            for k,v in d.items():
                prefix = k.split('/')[-1]
                handles.append( plt.plot(d[k]['accuracy'][:, 1], d[k]['accuracy'][:, 2], label=prefix)[0] )
                
                plt.title( prefix.split('-')[0] + '-lr')
                plt.ylabel('accuracy')
                plt.xlabel('sample')

                plt.legend(handles=handles, framealpha=0.5, frameon=False)
    plt.show(block=False)

    if savepath is not None:
        plt.savefig(savepath, format='eps', dpi=1000)
    


if __name__ == '__main__':
    basepath = '/home/christine/projects/convnet/paper/data/'
    savepath = '/home/christine/projects/convnet/paper/figures/'
    
    # nonlinear
    print('getting filenames...')
    abs_exper = sorted(glob(basepath + 'abs*'))
    relu_exper = sorted(glob(basepath + 'relu*'))
    select_exper = sorted(glob(basepath + 'select-*'))
    select_max_exper = sorted(glob(basepath + 'select_max*'))

    print('loading data...')
    abs_data = load_experiment_data(abs_exper)
    relu_data = load_experiment_data(relu_exper)
    select_data = load_experiment_data(select_exper)
    select_max_data = load_experiment_data(select_max_exper)

    print('displying data...')
    display_griddata([abs_data, relu_data, select_data, select_max_data], savepath + 'nonlinearity')
    
    # aggregation
    avg_exper = sorted(glob(basepath + 'avg*'))
    max_norm_exper = sorted(glob(basepath + 'max_norm*'))
    max_exper = sorted(glob(basepath + 'max-*'))
    
    avg_data = load_experiment_data(avg_exper)
    max_norm_data = load_experiment_data(max_norm_exper)
    max_data = load_experiment_data(max_exper)
    
    display_griddata([avg_data, max_data, max_norm_data], savepath + 'aggregation')

    plt.ginput()
