import h5py
import argparse
import numpy as np
import keras
import keras.backend as K
from keras.models import load_model
from utils import *
from keras.callbacks import ReduceLROnPlateau

# Command parser
parser = argparse.ArgumentParser(description='thicknessML transfer-learning')
parser.add_argument('--STL', action='store_true',
                    help='loading pre-trained Single-Task Learning thicknessML models (default: loading MultiTask Learning models)')
parser.add_argument('--full-weight', action='store_true',
                    help='enable full weight retraining (default: partial weight retraining by freezing feature extraction weights)')
args = parser.parse_args()

subset_no_MA = [8,15,9,12,11,16,13,10] # literature perovskite that contain no MA
random_states = (1, 362, 27, 81263, 516) # arbitrarily selected random states
mode = {
    True: 'STL',
    False: 'MTL',
    } # mode to be 'MTL' or 'STL' depending on args.STL
training_mode = {
    True: 'full weight',
    False: 'partial weight'
    } # training mode to be 'full weight' or 'partial weight' depending on args.full_weight

# Load experimental perovskite data to predict thickness
f = h5py.File("data/Expdataset.h5", "r")
X_exp = f['X_exp'][:]
d_exp = f['d_exp'][:]
f.close()

y_hat_list = []
for i in range(3): # load ensemble pre-trained models
    for j, random_state in enumerate(random_states):
        print(f'loading pre-trained {mode[args.STL]} model {i+1} out of 3; random state {j+1} out of 5')
        # Load retraining data
        X_train, y_train = get_retrainingdata(
            num_of_train=len(subset_no_MA), 
            whether_STL=args.STL,
            random_state=random_state, 
            subset_nk=subset_no_MA
        )
        
        losses = custom_objects(args.STL)
        
        K.clear_session()
        # Load pre-trained model
        model = load_model(f'pre-trained models/model_{mode[args.STL]}_{i}.h5',
                           compile=True,
                           custom_objects=losses)
        
        # Freeze feature extraction (convolutional layers) weights if it's partial-weight retraining
        if not args.full_weight:
            for layer in model.layers[:14]:
                layer.trainable = False
        
        if not args.STL:
            loss = [losses[k] for k in ('n_loss', 'k_loss', 'd_loss')]
        else:
            loss = losses['d_loss']
        
        # Retrain model
        model.compile(optimizer=keras.optimizers.Adagrad(lr=1e-3), loss=loss)
        RLR = ReduceLROnPlateau(monitor='loss', verbose=0, factor=0.2,
                                patience=10, min_lr=1e-4, min_delta=1e-4)
        print(f'{training_mode[args.full_weight]} retrianing')
        model.fit(x=X_train,
                  y=y_train,
                  batch_size=12,
                  epochs=200,
                  initial_epoch=0,
                  callbacks=[RLR],
                  shuffle=True
                  )
        
        print('predicting experimental thicknesses')
        # Predicted experimental thicknesses using retrained model
        y_hat = model.predict(X_exp, verbose=1)
        if args.STL:
            y_hat_list.append(y_hat)
        else:
            y_hat_list.append(y_hat[-1])

# Evaluate predicted thicknesses via plotting a figure similar to Figure 4
plot_exp(d_exp, np.hstack(y_hat_list))