# 为CNN寻找参数，包括学习率，动量参数，权重衰减参数
# 我们找两层的，由于MLP是两层的（隐藏层）
# An example of read in the data and train the model. The runner is implemented, while the model used for training need your implementation.
import mynn as nn
import numpy as np
from struct import unpack
import gzip
import pickle
import os

lr_list = [0.001, 0.005, 0.01, 0.02]
mu_list = [0.5, 0.9, 0.99]
wd_list = [1e-5, 1e-4]
channel_1 = [4, 8]
channel_2 = [4, 8]

# fixed seed for experiment
np.random.seed(309)

train_images_path = r'.\dataset\MNIST\train-images-idx3-ubyte.gz'
train_labels_path = r'.\dataset\MNIST\train-labels-idx1-ubyte.gz'

with gzip.open(train_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        train_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
with gzip.open(train_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        train_labs = np.frombuffer(f.read(), dtype=np.uint8)


# choose 10000 samples from train set as validation set.
idx = np.random.permutation(np.arange(num))
# save the index.
with open('idx.pickle', 'wb') as f:
        pickle.dump(idx, f)
train_imgs = train_imgs[idx]
train_labs = train_labs[idx]
valid_imgs = train_imgs[:10000]
valid_labs = train_labs[:10000]
train_imgs = train_imgs[10000:]
train_labs = train_labs[10000:]

# normalize from [0, 255] to [0, 1]
train_imgs = train_imgs / train_imgs.max()
valid_imgs = valid_imgs / valid_imgs.max()

train_imgs = train_imgs / train_imgs.max()
valid_imgs = valid_imgs / valid_imgs.max()

train_imgs = train_imgs.reshape(-1, 1, 28, 28)
valid_imgs = valid_imgs.reshape(-1, 1, 28, 28)

class RunnerM4search():
    """
    This is an exmaple to train, evaluate, save, load the model. However, some of the function calling may not be correct 
    due to the different implementation of those models.
    """
    def __init__(self, model, optimizer, metric, loss_fn, batch_size=32, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric = metric
        self.scheduler = scheduler
        self.batch_size = batch_size

        self.train_scores = []
        self.dev_scores = []
        self.train_loss = []
        self.dev_loss = []

    def train(self, train_set, dev_set, max_iter=300, **kwargs):

        num_epochs = kwargs.get("num_epochs", 0)
        log_iters = kwargs.get("log_iters", 100)
        save_dir = kwargs.get("save_dir", "best_model")

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        best_score = 0

        for epoch in range(num_epochs):
            X, y = train_set

            assert X.shape[0] == y.shape[0]

            idx = np.random.permutation(range(X.shape[0]))

            X = X[idx]
            y = y[idx]

            for iteration in range(min(int(X.shape[0] / self.batch_size) + 1, max_iter)):
                train_X = X[iteration * self.batch_size : (iteration+1) * self.batch_size]
                train_y = y[iteration * self.batch_size : (iteration+1) * self.batch_size]

                logits = self.model(train_X)
                trn_loss = self.loss_fn(logits, train_y)
                self.train_loss.append(trn_loss)
                
                trn_score = self.metric(logits, train_y)
                self.train_scores.append(trn_score)

                # the loss_fn layer will propagate the gradients.
                self.loss_fn.backward()

                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                
                dev_score, dev_loss = self.evaluate(dev_set)
                self.dev_scores.append(dev_score)
                self.dev_loss.append(dev_loss)

                """if (iteration) % log_iters == 0:
                    print(f"epoch: {epoch}, iteration: {iteration}")
                    print(f"[Train] loss: {trn_loss}, score: {trn_score}")
                    print(f"[Dev] loss: {dev_loss}, score: {dev_score}")"""

            if dev_score > best_score:
                # save_path = os.path.join(save_dir, 'best_model.pickle')
                print(f"best accuracy performence has been updated: {best_score:.5f} --> {dev_score:.5f}")
                best_score = dev_score
        self.best_score = best_score

    def evaluate(self, data_set):
        X, y = data_set
        logits = self.model(X)
        loss = self.loss_fn(logits, y)
        score = self.metric(logits, y)
        return score, loss


import itertools
best_score = 0
best_loss = 0
for lr, mu, wd, c1, c2 in itertools.product(lr_list, mu_list, wd_list, channel_1, channel_2):
    model = nn.models.Model_CNN(
        channels_list=[c1, c2],kernel_size_list=[3, 3],stride_list=[1, 1],
        num_classes=10,image_size=28,act_func='ReLU',weight_decay_lambda=wd
    )
    optimizer = nn.optimizer.MomentGD(init_lr=lr, model=model, mu=mu)
    scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, gamma=0.2, milestones=[290])
    loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=train_labs.max()+1)

    runner = RunnerM4search(model, optimizer, nn.metric.accuracy, loss_fn, scheduler=scheduler)

    runner.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=1, log_iters=100)
    
    current_best = runner.best_score
    if current_best > best_score:
        best_score = current_best
        best_params = {'lr': lr, 'mu': mu, 'wd': wd, 'channels': [c1, c2]}
        print(f"已寻找到一个更合适的超参:score={best_score:.4f}, params={best_params}")
