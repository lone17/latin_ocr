from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from model_utils import *
from models import *

cb_list = []

es = EarlyStopping(monitor='loss', patience=4, mode='min', verbose=1)
cb_list.append(es)
# cp = ModelCheckpoint(filepath='checkpoints/{val_loss:.2f}.model_',
#                      monitor='val_loss', verbose=1, mode='min', save_best_only=True)
# cb_list.append(cp)

# model.load_weights('checkpoints/17.67.model_')
gen = DataGenerator('X_real', 'y_real', down_width_factor['model_'], 8)
import tensorflow as tf
with tf.device('/gpu:0'):
    model.fit_generator(generator=gen.next_train(),
                        steps_per_epoch=gen.train_steps,
                        validation_data=gen.next_val(),
                        validation_steps=gen.val_steps,
                        epochs=100,
                        callbacks=cb_list,
                        verbose=1,
                        )
