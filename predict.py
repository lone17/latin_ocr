import json
import numpy as np
from Levenshtein import distance
from models import *
from text_utils import *
from image_utils import *

model_name = 'model7'
weights_dir = 'checkpoints/8.08.' + model_name
data_dir = 'real/'
model = models[model_name](training=False)
model.load_weights(weights_dir)

def predict(img_dir):
    img = preprocess(img_dir, padding=True)
    img = np.expand_dims(img, 0)
    pred = model.predict(img, batch_size=1)

    return ctc_decoder(pred)

# f = open(os.path.join(data_dir, 'labels.json'), encoding='utf-8')
# files = list(json.load(f).keys())
# f.close()
permu = np.load('permu.npy')[-100:]
with open('transcripts_real.txt', 'r', encoding='utf-8') as f:
    transcripts = f.read().split('\n')
transcripts = [transcripts[i] for i in permu]
with open('images_real.txt', 'r') as f:
    files = f.read().split('\n')
files = [files[i] for i in permu]
X = np.load('X_real.npy', mmap_mode='r')[permu]
# y_pred = model.predict(X, batch_size=16)

losses = []
# for i in permu[-2500:]:
for img, label, data in zip(files, transcripts, X):
    # img = files[i]
    # pred = predict(os.path.join(data_dir, img))
    # pred = np.expand_dims(pred, 0)
    data = np.expand_dims(data, 0)
    pred = model.predict(data, batch_size=1)
    pred = ctc_decoder(pred)
    loss = distance(pred, label) / max(len(pred), len(label))
    losses.append(loss)
    print(img, loss, '= \'' + pred + '\'')

print(np.mean(losses))
