import math
import random
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

down_width_factor = {
    'model0': 3*2*2*2,
    'model1': 3*2*2*2,
    'model2': 3*2*2*2,
    'model3': 3*2*2*2,
    'model4': 2*2*2*2,
    'model5': 2*2*2*1,
    'model6': 2*2*2*1,
    'model7': 2*2*2*1,
    'model_': 2*2*2*1,
}

down_height_factor = {
    'model0': 2*2*2*2,
    'model1': 2*2*2*2,
    'model2': 2*2*2*2,
    'model3': 2*2*2*2,
    'model4': 2*2*2*2,
    'model5': 2*2*2*2,
    'model6': 2*2*2*2,
    'model7': 2*2*2*2,
    'model_': 2*2*2*2,
}

class DataGenerator:

    def __init__(self, X_file, y_file, down_width_factor, batch_size=32, val_size=0.25):
        X = np.load(X_file + '.npy', mmap_mode='r')
        y = np.load(y_file + '.npy')
        permu = list(range(len(y)))
        random.shuffle(permu)

        # self.X_train, self.X_val, self.y_train, self.y_val = \
        # train_test_split(X, y, test_size=val_size, shuffle=True)
        split = int(len(permu) * val_size)
        train_idx = permu[:-split]
        val_idx = permu[-split:-100]
        self.X_train = X[train_idx]
        self.X_val = X[val_idx]
        self.y_train = y[train_idx]
        self.y_val = y[val_idx]

        self.input_length = X.shape[1] // down_width_factor - 2
        del X
        del y

        # self.down_width_factor = down_width_factor
        self.batch_size = batch_size
        self.train_size = len(self.X_train)
        self.val_size = len(self.X_val)
        self.train_steps = math.ceil(self.train_size / batch_size)
        self.val_steps = math.ceil(self.val_size / batch_size)

    def next_train(self):
        while True:
            for i in range(0, self.train_size, self.batch_size):
                X = self.X_train[i : i+self.batch_size]
                y = self.y_train[i : i+self.batch_size]

                batch_size = len(X)

                input_length = np.ones([batch_size, 1]) * self.input_length
                label_length = np.array([sum(label != -1) for label in y])[:None]

                inputs = {
                    'input': X,
                    'labels': y,
                    'input_length': input_length,
                    'label_length': label_length,
                    }

                outputs = {'ctc': np.zeros([batch_size])}

                yield (inputs, outputs)

    def next_val(self):
        while True:
            for i in range(0, self.val_size, self.batch_size):
                X = self.X_val[i : i+self.batch_size]
                y = self.y_val[i : i+self.batch_size]

                batch_size = len(X)

                input_length = np.ones([batch_size, 1]) * self.input_length
                label_length = np.array([sum(label != -1) for label in y])[:None]

                inputs = {
                    'input': X,
                    'labels': y,
                    'input_length': input_length,
                    'label_length': label_length,
                    }

                outputs = {'ctc': np.zeros([batch_size])}

                yield (inputs, outputs)
