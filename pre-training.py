import h5py, os
import argparse
import numpy as np
import keras
import keras.backend as K
from keras.layers import Conv1D, MaxPooling1D, Flatten, Activation, Dense, Dropout, Input
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
from keras.models import load_model, save_model, Model

# Command parser
parser = argparse.ArgumentParser(description='thicknessML pre-training')
parser.add_argument('--STL', action='store_true',
                    help='training Single-Task Learning thicknessML models (default: training MultiTask Learning models)')
args = parser.parse_args()

mode = {
        True: 'STL',
        False: 'MTL',
        } # mode to be 'MTL' or 'STL' depending on args.STL

for i in range(3):
    
    # Read pre-training TL datasets (multiple datasets are for ensemble training)
    f = h5py.File(f"data/TLdataset_{i}.h5", "r")
    keys = ('X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test')
    [X_train, X_val, X_test, y_train, y_val, y_test] = [f[key][:] for key in keys]
    f.close()
    
    if not args.STL:
        y_train = [y_train[:,:651], y_train[:,651:-1], y_train[:,-1:]] # list of R, T, d
        y_val = [y_val[:,:651], y_val[:,651:-1], y_val[:,-1:]]
        y_test = [y_test[:,:651], y_test[:,651:-1], y_test[:,-1:]]
    else:
        y_train = y_train[:,-1:]
        y_val = y_val[:,-1:]
        y_test = y_test[:,-1:]
    
    # Build and train thicknessML model
    K.clear_session()
    
    model_input = Input(shape=(651,2))
    shared = Conv1D(512,8,strides=1,padding='valid')(model_input)
    shared = Activation("relu")(shared)
    shared = MaxPooling1D(3)(shared)
    shared = Conv1D(128,5,strides=1,padding='valid')(shared)
    shared = Activation("relu")(shared)
    shared = MaxPooling1D(3)(shared)
    shared = Conv1D(64,3,strides=1,padding='valid')(shared)
    shared = Activation("relu")(shared)
    shared = MaxPooling1D(2)(shared)
    shared = Conv1D(32,3,strides=1,padding='valid')(shared)
    shared = Activation("relu")(shared)
    shared = MaxPooling1D(2)(shared)
    shared = Flatten()(shared)
    
    if not args.STL:
        # Multitask: infer n
        y1 = Dense(2048, activation='relu')(shared)
        y1 = Dropout(0.3)(y1)
        y1 = Dense(1024, activation='relu')(y1)
        y1 = Dropout(0.3)(y1)
        y1 = Dense(651, activation='relu', name='n')(y1)
        
        # Multitask: infer k
        y2 = Dense(2048, activation='relu')(shared)
        y2 = Dropout(0.3)(y2)
        y2 = Dense(1024, activation='relu')(y2)
        y2 = Dropout(0.3)(y2)
        y2 = Dense(651, activation='relu', name='k')(y2)
    
    # infer d
    y3 = Dense(2048, activation='relu')(shared)
    y3 = Dropout(0.3)(y3)
    y3 = Dense(1024, activation='relu')(y3)
    y3 = Dropout(0.3)(y3)
    y3 = Dense(512, activation='relu')(y3)
    y3 = Dropout(0.3)(y3)
    y3 = Dense(1, activation='relu', name='d')(y3)
    
    if not args.STL:
        model = Model(inputs=model_input, outputs=[y1, y2, y3])
    else:
        model = Model(inputs=model_input, outputs=y3)
    
    # logcosh for buidling loss function(s)
    def _logcosh(x):
        return x + K.softplus(-2. * x) - K.log(2.)
    
    if not args.STL:
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
    
    if not args.STL:
        model.compile(optimizer=keras.optimizers.Adagrad(), loss=[n_loss, k_loss, d_loss])
        RLR = ReduceLROnPlateau(monitor='val_loss', verbose=1,
                                patience=10, min_lr=0, min_delta=1e-4)
        def scheduler(epoch, lr):
            if epoch%50 == 0 and epoch>=150:
                lr = 1e-3
            return lr
        LRS = LearningRateScheduler(scheduler, verbose=1)
    else:
        model.compile(optimizer=keras.optimizers.Adagrad(), loss=d_loss)
        RLR = ReduceLROnPlateau(monitor='val_loss', verbose=1, factor=0.2,
                                patience=10, min_lr=1e-4, min_delta=1e-4)
        def scheduler(epoch, lr):
            if epoch%50 == 0 and epoch>=150:
                lr = 1e-2
            return lr
        LRS = LearningRateScheduler(scheduler, verbose=1)
    
    print(f'pre-training {i+1} out of 3 {mode[args.STL]} models')
    
    model.fit(x=X_train,
              y=y_train,
              batch_size=128,
              epochs=2000,
              validation_data=(X_val, y_val),
              initial_epoch=0,
              callbacks=[RLR, LRS],
              shuffle='batch'
              )
    
    os.makedirs('pre-trained models', exist_ok=True)
    
    print(f'saving {i+1} out of 3 pre-trained {mode[args.STL]} models')
    save_model(model,
               f'pre-trained models/model_{mode[args.STL]}_{i}.h5',
               overwrite = True,
               include_optimizer = True
               )