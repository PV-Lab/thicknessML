import h5py
import numpy as np
import keras.backend as K
import matplotlib.pyplot as plt
import matplotlib as mpl

def get_retrainingdata(num_of_train, random_state, whether_STL=False, 
                       training_thickness_num=10, subset_nk=None):
    '''
    get_data obtains data for retraining.

    Parameters
    ----------
    num_of_train : int
        Number of training nk spectra (out of 18 literature perovskite nk), 
        and the rest are automatically test nk spectra.
    random_state : int
        Random seed for generating the training/test set.
    whether_STL : boolean
        True for loading STL data, and False for loading MTL data. Default is False.
    training_thickness_num : int, optional
        Number of thicknesses per training nk spectra used for RT simulation. 
        The default is 10 for retraining.
    subset_nk : list, optional
        List of selected (subset of) training nk for data generation. 
        The default is None. Used to only select literature nk spectra of perovskites
        that don't contain methylammonium (MA) as training data to predict 
        experimental MAPbI3 films. When inputted, only X_train and y_train are outputted.

    Returns
    -------
    X_train : numpy ndarray
        training set input.
    X_test : numpy ndarray
        test set input.
    y_train : numpy ndarray
        training set output.
    y_test : numpy ndarray
        test set output.

    '''
    
    # Load retraining data, where Perovdataset.h5 contains RT simulation for all
    # thickness 10—2010 nm for every literature perovskite nk spectra
    f = h5py.File("data/Perovdataset.h5", "r")
    n_list = f['n']
    k_list = f['k']
    d_list = f['d']
    R_list = f['R']
    T_list = f['T']
    
    # Allocate training and test index
    np.random.seed(random_state)
    if not subset_nk:
        ind = np.arange(n_list.shape[0])
    else:
        ind = subset_nk
    np.random.shuffle(ind)
    ind_train = ind[:num_of_train]
    if not subset_nk:
        ind_test = np.setdiff1d(ind,ind_train)
    
    # Generate training data
    X_train = np.empty((1, 651, 2))
    y_train = np.empty((1, 1303))
    
    for counter, i in enumerate(np.sort(ind_train)):
        np.random.seed(random_state+1+counter)
        ind = np.random.choice(np.arange(2001), training_thickness_num, replace=False)
        R = np.vstack((R_list[np.sort(ind)+i*2001,:]))
        T = np.vstack((T_list[np.sort(ind)+i*2001,:]))
        RT = np.array([R,T]).astype('float16')
        RT = np.rollaxis(RT, 0, 3)
        X_train = np.append(X_train, RT, axis=0)
        
        del R, T, RT
        
        n = np.tile(n_list[i,:][np.newaxis,:], (training_thickness_num,1))
        k = np.tile(k_list[i,:][np.newaxis,:], (training_thickness_num,1))
        d = d_list[np.sort(ind)]
        nkd = np.hstack((n,k,d))
        y_train = np.append(y_train, nkd, axis=0)
        
        del nkd
        
    X_train = np.delete(X_train, 0, axis=0)
    y_train = np.delete(y_train, 0, axis=0)
    y_train = [y_train[:,:651], y_train[:,651:-1], y_train[:,-1][:,np.newaxis]]
    
    if not subset_nk:
        # Generate test data
        test_thickness_num = 50
        X_test = np.empty((1, 651, 2))
        y_test = np.empty((1, 1303))
        
        for counter, i in enumerate(np.sort(ind_test)):
            np.random.seed(random_state+2+counter)
            ind = np.random.choice(np.arange(2001), test_thickness_num, replace=False)
            R = np.vstack((R_list[np.sort(ind)+i*2001,50:701]))
            T = np.vstack((T_list[np.sort(ind)+i*2001,50:701]))
            RT = np.array([R,T]).astype('float16')
            RT = np.rollaxis(RT, 0, 3)
            X_test = np.append(X_test, RT, axis=0)
            
            del R, T, RT
            
            n = np.tile(n_list[i,:][np.newaxis,:], (test_thickness_num,1))
            k = np.tile(k_list[i,:][np.newaxis,:], (test_thickness_num,1))
            d = d_list[np.sort(ind)]
            nkd = np.hstack((n,k,d))
            y_test = np.append(y_test, nkd, axis=0)
            
            del nkd
            
        X_test = np.delete(X_test, 0, axis=0)
        y_test = np.delete(y_test, 0, axis=0)
        y_test = [y_test[:,:651], y_test[:,651:-1], y_test[:,-1][:,np.newaxis]]
    f.close()
    
    if whether_STL:
        y_train = y_train[-1]
        try:
            y_test = y_test[-1]
        except:
            pass
    
    if subset_nk:
        return X_train, y_train
    else:
        return X_train, X_test, y_train, y_test

def custom_objects(whether_STL=False):
    '''
    Load custom_objects (custom loss functions in pre-training) needed for 
    loading pre-training models

    Parameters
    ----------
    whether_STL : boolean
        True for loading STL models, and False for loading MTL models. Default is False.

    Returns
    -------
    custom_objects: dict
        custom_objects dictionary to be fed to keras.models.load_model.

    '''
    
    def _logcosh(x):
        return x + K.softplus(-2. * x) - K.log(2.)
    
    if not whether_STL:
        def n_loss(y_true, y_pred):
            '''
            n_loss places fractionally higher weight on smaller n values
            assuming a range from 0 to 10, i.e., the calculated loss for n=10 
            is multiplied by 1, and the calculated loss for n=0 is multiplied by 2.
            This is parameterized by unit_weights, mag, and bias, where unit_weights
            sets the fractional range, mag and bias co-determine the scaling multiplier.
            '''
            unit_weights = (K.ones(K.shape(y_pred))*10 - y_true)/10
            bias = 1
            mag = 1
            weights = mag*unit_weights + bias
            loss = 10*K.mean(_logcosh(y_pred - y_true)*weights, axis=-1)
            return loss
        
        def k_loss(y_true, y_pred):
            '''
            Fractional weights not implemented because of many near zero k.
            '''
            loss = 8*K.mean(_logcosh(y_pred - y_true), axis=-1)
            return loss
    
    def d_loss(y_true, y_pred):
        '''
        d_loss places fractionally higher weight on smaller d values
        assuming a range from 0 to 2010 nm, i.e., the calculated loss for d=2010 
        is multiplied by 1, and the calculated loss for d=0 is multiplied by 2.5.
        This is parameterized by unit_weights, mag, and bias, where unit_weights
        sets the fractional range, mag and bias co-determine the scaling multiplier.
        '''
        unit_weights = (K.ones(K.shape(y_pred))*2010 - y_true)/2010
        bias = 1
        mag = 1.5
        weights = mag*unit_weights + bias
        loss = 0.01*K.mean(_logcosh(y_pred - y_true)*weights, axis=-1)
        return loss
    
    if not whether_STL:
        return {'n_loss': n_loss,
                'k_loss': k_loss,
                'd_loss': d_loss}
    else:
        return {'d_loss': d_loss}

def mean_absolute_percentage_error(y_true, y_pred, axis=None):
    return np.mean(np.abs((y_true - y_pred) / y_true), axis=axis)

def plot_exp(y, y_hat):
    '''
    plot function for experimental thickness prediction

    Parameters
    ----------
    y : numpy ndarray
        measured thickness.
    y_hat_mean : numpy ndarray
        predicted thickensses (of multiple runs from ensemble models and training data splits).
    '''
    
    MAPE = mean_absolute_percentage_error(y, y_hat, axis=1) # MAPE averaged over ensemble models and training data splits
    # print('MAPE min, max: %.0f%%, %.0f%%' % (min(MAPE)*100, max(MAPE)*100))
    y_hat_mean = np.mean(y_hat, axis=1)
    y_hat_std = np.std(y_hat, axis=1)

    
    math_font = 'cm'
    mpl.style.use('default')
    font = {
            'family': 'Calibri',
            'size': 30,
            'weight': 'light'
        }
    
    math_font = 'cm'
    
    mpl.rc('font', **font)
    mpl.rcParams['mathtext.fontset'] = math_font
    mpl.rcParams['axes.labelweight'] = 'light'
    mpl.rcParams['xtick.labelsize'] = font['size']-4
    mpl.rcParams['ytick.labelsize'] = font['size']-4
    mpl.rcParams['axes.labelsize'] = font['size']-1
    mpl.rcParams['axes.titleweight'] = font['weight']
    mpl.rcParams['figure.figsize'] = [8, 5]
    mpl.rcParams['legend.fontsize'] = 27
    mpl.rcParams['legend.frameon'] = True
    mpl.rcParams['figure.dpi'] = 50
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 7.1))
    ax.scatter(y, y_hat_mean, 30, c='#e31a1c')
    ax.errorbar(y, y_hat_mean, yerr=y_hat_std, fmt='none', ecolor='#e31a1c', elinewidth=1.5, capsize=3, capthick=1.5)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
    ax.plot([y.min(), y.max()], [1.1*y.min(), 1.1*y.max()], 'k--', lw=2)
    ax.plot([y.min(), y.max()], [0.9*y.min(), 0.9*y.max()], 'k--', lw=2)
    ax.set_xlabel('Actual $d$ [nm]')
    ax.set_ylabel('Predicted $d$ [nm]')
    ax.tick_params(axis='both', which='major')
    ax.set_title('MAPE min, max: %.0f%%, %.0f%%' % (min(MAPE)*100, max(MAPE)*100))
    plt.show()