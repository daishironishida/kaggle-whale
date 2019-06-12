import gzip
import base64
import os
from pathlib import Path
from typing import Dict

n_epochs = 20
image_size = 288
batch_size = 32
step_size = 1
lr = 1e-4
dropout = 0
patience = 2
model = "resnet50"
loss = "bce"
transform = "original"
optim = "adam"
scheduler = "none"
schedule_length = 0
prev_model = "none"

# this is base64 encoded source code
file_data: Dict = {file_data}


for path, encoded in file_data.items():
    print(path)
    path = Path(path)
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(gzip.decompress(base64.b64decode(encoded)))


def run(command):
    os.system('export PYTHONPATH=${PYTHONPATH}:/kaggle/working && ' + command)


run('python setup.py develop --install-dir /kaggle/working')
run('python -m whale.train_test_split')
run('python -m whale.main train model_1 --model {} --dropout {} --image-size {} --batch-size {} --step {} --n-epochs {} --patience {} --lr {} --loss {} --transform {} --optim {} --scheduler {} --schedule-length {} --prev-model {}'.format(
    model, dropout, image_size, batch_size, step_size, n_epochs, patience, lr, loss, transform, optim, scheduler, schedule_length, prev_model))
run('python -m whale.main predict_test model_1 --model {} --dropout {} --image-size {} --batch-size {} --transform {}'.format(
    model, dropout, image_size, batch_size, transform))
run('python -m whale.make_submission model_1/test.h5 submission.csv')
