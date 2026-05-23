import gzip
import pickle
import os
from struct import unpack
import matplotlib.pyplot as plt
import numpy as np
import mynn as nn

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def load_mnist(image_path, label_path, flatten=False, limit=None):
    with gzip.open(image_path, 'rb') as f:
        _, num, rows, cols = unpack('>4I', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)

    with gzip.open(label_path, 'rb') as f:
        _, num = unpack('>2I', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    images = (images / 255.0).astype(np.float32)
    if limit is not None:
        images = images[:limit]
        labels = labels[:limit]
    if flatten:
        images = images.reshape(images.shape[0], -1)
    else:
        images = images[:, None, :, :]
    return images, labels


def softmax(logits):
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def infer_model_type(model_path):
    with open(model_path, 'rb') as f:
        param_list = pickle.load(f)
    return 'cnn' if isinstance(param_list[0], dict) else 'mlp'

def matrix_metrics(matrix):
    matrix = np.asarray(matrix)
    total = matrix.sum()
    tp = np.diag(matrix)
    pred_sum = matrix.sum(axis=0)
    true_sum = matrix.sum(axis=1)

    precision = np.divide(tp, pred_sum, out=np.zeros_like(tp, dtype=np.float64), where=pred_sum != 0)
    recall = np.divide(tp, true_sum, out=np.zeros_like(tp, dtype=np.float64), where=true_sum != 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(precision, dtype=np.float64),
                   where=(precision + recall) != 0)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def save_metrics_csv(matrix, save_path):
    metrics = matrix_metrics(matrix)
    parent_dir = os.path.dirname(save_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('class,precision,recall,f1_score\n')
        for i in range(matrix.shape[0]):
            f.write(
                f"{i},{metrics['precision'][i]:.6f},"
                f"{metrics['recall'][i]:.6f},"
                f"{metrics['f1'][i]:.6f}\n"
            )
    return metrics


class report_visualization():
    def __init__(self, model_path, dataset, num_classes=10, model_type='auto',
                 limit=None, save_prefix=None) -> None:
        self.model_path = model_path
        self.dataset = dataset
        self.num_classes = num_classes
        self.model_type = model_type
        self.limit = limit
        self.save_prefix = save_prefix

        self.model = None
        self.images = None
        self.labels = None
        self.logits = None
        self.probs = None
        self.preds = None

    def __call__(self):
        self.load()
        matrix = self.confusion_matrix()
        self.show()
        return matrix

    def load(self):
        model_type = infer_model_type(self.model_path) if self.model_type == 'auto' else self.model_type
        if model_type == 'cnn':
            self.model = nn.models.Model_CNN()
        elif model_type == 'mlp':
            self.model = nn.models.Model_MLP()
        else:
            raise ValueError("model_type should be 'auto', 'cnn', or 'mlp'")
        self.model.load_model(self.model_path)
        is_cnn = isinstance(self.model, nn.models.Model_CNN)

        if self.dataset is not None:
            self.images, self.labels = self.dataset
            if not is_cnn and self.images.ndim > 2:
                self.images = self.images.reshape(self.images.shape[0], -1)
        else:
            raise ValueError('dataset should be offered as (images, labels).')

        self.logits = self.model(self.images)
        self.probs = softmax(self.logits)
        self.preds = np.argmax(self.logits, axis=1)

    def confusion_matrix(self):
        # 混淆矩阵绘制
        if self.preds is None:
            self.load()

        matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        for true_label, pred_label in zip(self.labels, self.preds):
            matrix[true_label, pred_label] += 1

        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(matrix, cmap='Blues')
        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True label')
        ax.set_title('Confusion Matrix')
        ax.set_xticks(np.arange(self.num_classes))
        ax.set_yticks(np.arange(self.num_classes))

        threshold = matrix.max() * 0.6
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                color = 'white' if matrix[i, j] > threshold else 'black'
                ax.text(j, i, str(matrix[i, j]), ha='center', va='center', color=color, fontsize=8)

        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        if self.save_prefix is not None:
            save_dir = os.path.dirname(self.save_prefix)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            fig.savefig(f'{self.save_prefix}_confusion_matrix.png', dpi=160)
            save_metrics_csv(matrix, f'{self.save_prefix}_metrics.csv')
        plt.show()
        return matrix

    def show(self, correct_num=6, wrong_num=6):
        if self.preds is None:
            self.load()

        correct_indices = np.where(self.preds == self.labels)[0]
        wrong_indices = np.where(self.preds != self.labels)[0]

        rng = np.random.default_rng(0)
        if correct_indices.size > correct_num:
            correct_indices = rng.choice(correct_indices, size=correct_num, replace=False)

        # 选取交叉熵loss最大的几个展示
        if wrong_indices.size > wrong_num:
            losses = -np.log(self.probs[wrong_indices, self.labels[wrong_indices]] + 1e-10)
            wrong_indices = wrong_indices[np.argsort(losses)[-wrong_num:]]

        self._plot_examples(correct_indices, 'Correct Predictions', 'correct')
        self._plot_examples(wrong_indices, 'Highest Loss Mistakes', 'wrong')

    def _plot_examples(self, indices, title, suffix):
        if len(indices) == 0:
            return

        fig, axes = plt.subplots(len(indices), 2, figsize=(7, 2.1 * len(indices)))
        if len(indices) == 1:
            axes = axes[None, :]

        for row, idx in enumerate(indices):
            image = self.images[idx]
            if image.ndim == 3:
                image = image[0]
            else:
                image = image.reshape(28, 28)

            axes[row, 0].imshow(image, cmap='gray')
            axes[row, 0].set_title(f'true={self.labels[idx]}, pred={self.preds[idx]}')
            axes[row, 0].axis('off')

            axes[row, 1].bar(np.arange(self.num_classes), self.probs[idx])
            axes[row, 1].set_ylim(0, 1)
            axes[row, 1].set_xticks(np.arange(self.num_classes))
            axes[row, 1].set_ylabel('prob')

        fig.suptitle(title)
        fig.tight_layout()
        if self.save_prefix is not None:
            fig.savefig(f'{self.save_prefix}_{suffix}_examples.png', dpi=160)
        plt.show()
