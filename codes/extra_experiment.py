# Part C : 可视化和鲁棒性测试
import os

import matplotlib.pyplot as plt
import numpy as np

import mynn as nn
from visualization import load_mnist, report_visualization
from weight_visualization import visualize_baseline_cnn_weights


plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

SAVED_MODELS_DIR = r'.\saved_models'
EXTRA_RESULTS_DIR = r'.\extra_results'

MLP_MODEL_PATH = r'.\saved_models\best_model_MLP.pickle'
CNN_MODEL_PATH = r'.\saved_models\best_model_CNN.pickle'

train_images_path = r'.\dataset\MNIST\train-images-idx3-ubyte.gz'
train_labels_path = r'.\dataset\MNIST\train-labels-idx1-ubyte.gz'


def load_saved_models():
    mlp = nn.models.Model_MLP()
    mlp.load_model(MLP_MODEL_PATH)

    cnn = nn.models.Model_CNN()
    cnn.load_model(CNN_MODEL_PATH)
    return mlp, cnn


def predict_in_batches(model, images_nchw, batch_size=128):
    logits = []
    is_cnn = isinstance(model, nn.models.Model_CNN)
    for start in range(0, images_nchw.shape[0], batch_size):
        batch = images_nchw[start:start + batch_size]
        if not is_cnn:
            batch = batch.reshape(batch.shape[0], -1)
        logits.append(model(batch))
    return np.concatenate(logits, axis=0)


def accuracy_from_logits(logits, labels):
    preds = np.argmax(logits, axis=1)
    return (preds == labels).sum() / labels.shape[0], preds


def plot_confusion_matrix(labels, preds, title, save_path):
    matrix = np.zeros((10, 10), dtype=np.int64)
    for true_label, pred_label in zip(labels, preds):
        matrix[true_label, pred_label] += 1

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap='Blues')
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title(title)
    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(10))

    threshold = matrix.max() * 0.6
    for i in range(10):
        for j in range(10):
            color = 'white' if matrix[i, j] > threshold else 'black'
            ax.text(j, i, str(matrix[i, j]), ha='center', va='center', color=color, fontsize=8)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.show()
    return matrix


def plot_digit_group_confusion_matrix(labels, preds, digit_group, title, save_path):
    digit_group = list(digit_group)
    matrix = np.zeros((len(digit_group), len(digit_group)), dtype=np.int64)
    label_to_idx = {label: idx for idx, label in enumerate(digit_group)}

    mask = np.isin(labels, digit_group) & np.isin(preds, digit_group)
    for true_label, pred_label in zip(labels[mask], preds[mask]):
        matrix[label_to_idx[true_label], label_to_idx[pred_label]] += 1

    fig, ax = plt.subplots(figsize=(4, 3.5))
    im = ax.imshow(matrix, cmap='Blues')
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title(title)
    ax.set_xticks(np.arange(len(digit_group)))
    ax.set_yticks(np.arange(len(digit_group)))
    ax.set_xticklabels(digit_group)
    ax.set_yticklabels(digit_group)

    threshold = matrix.max() * 0.6 if matrix.max() > 0 else 0
    for i in range(len(digit_group)):
        for j in range(len(digit_group)):
            color = 'white' if matrix[i, j] > threshold else 'black'
            ax.text(j, i, str(matrix[i, j]), ha='center', va='center', color=color, fontsize=10)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.show()
    return matrix


def evaluate_on_images(model, images_nchw, labels, name, save_confusion=True):
    logits = predict_in_batches(model, images_nchw)
    acc, preds = accuracy_from_logits(logits, labels)
    print(f'{name}: accuracy={acc:.4f}')

    matrix = None
    if save_confusion:
        matrix = plot_confusion_matrix(
            labels,
            preds,
            title=f'{name} Confusion Matrix',
            save_path=os.path.join(EXTRA_RESULTS_DIR, f'{name}_confusion_matrix.png'),
        )
    return acc, preds, matrix


def visualize_saved_models():
    report_visualization(
        model_path=MLP_MODEL_PATH,
        dataset_path=(train_images_path, train_labels_path),
        model_type='mlp',
        save_prefix=os.path.join(EXTRA_RESULTS_DIR, 'MLP_original'),
    )()

    report_visualization(
        model_path=CNN_MODEL_PATH,
        dataset_path=(train_images_path, train_labels_path),
        model_type='cnn',
        save_prefix=os.path.join(EXTRA_RESULTS_DIR, 'CNN_original'),
    )()


def visualize_cnn_weights():
    visualize_baseline_cnn_weights(
        CNN_MODEL_PATH,
        save_prefix=os.path.join(EXTRA_RESULTS_DIR, 'CNN_weights'),
        show=True,
    )


def gaussian_noise(images_nchw, sigma=0.15, seed=309):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, size=images_nchw.shape).astype(np.float32)
    return np.clip(images_nchw + noise, 0.0, 1.0)


def rotate_images(images_nchw, angle):
    rotated = np.empty_like(images_nchw)
    for idx, image in enumerate(images_nchw):
        rotated[idx] = rotate_single_image(image, angle)
    return rotated


def rotate_single_image(image, angle):
    c, h, w = image.shape
    rad = np.deg2rad(angle)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0

    yy, xx = np.indices((h, w))
    x0 = xx - cx
    y0 = yy - cy
    src_x = cos_a * x0 + sin_a * y0 + cx
    src_y = -sin_a * x0 + cos_a * y0 + cy

    src_x = np.rint(src_x).astype(np.int64)
    src_y = np.rint(src_y).astype(np.int64)
    valid = (src_x >= 0) & (src_x < w) & (src_y >= 0) & (src_y < h)

    output = np.zeros_like(image)
    for channel in range(c):
        output[channel, valid] = image[channel, src_y[valid], src_x[valid]]
    return output


def scale_images(images_nchw, scale):
    scaled = np.empty_like(images_nchw)
    for idx, image in enumerate(images_nchw):
        scaled[idx] = scale_single_image(image, scale)
    return scaled


def scale_single_image(image, scale):
    c, h, w = image.shape
    output = np.zeros_like(image)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0

    yy, xx = np.indices((h, w))
    src_x = (xx - cx) / scale + cx
    src_y = (yy - cy) / scale + cy

    src_x = np.rint(src_x).astype(np.int64)
    src_y = np.rint(src_y).astype(np.int64)
    valid = (src_x >= 0) & (src_x < w) & (src_y >= 0) & (src_y < h)

    for channel in range(c):
        output[channel, valid] = image[channel, src_y[valid], src_x[valid]]
    return output


def plot_reserved_digit_confusion_matrices(labels, preds, digit_groups):
    matrices = {}
    for digit_group in digit_groups:
        group_name = '_'.join(str(digit) for digit in digit_group)
        matrices[group_name] = plot_digit_group_confusion_matrix(
            labels,
            preds,
            digit_group=digit_group,
            title=f'Rotation 180 Confusion: {digit_group}',
            save_path=os.path.join(EXTRA_RESULTS_DIR, f'CNN_rotation_180_group_{group_name}_confusion_matrix.png'),
        )
    return matrices


def robustness_tests(mlp, cnn):
    train_images_path = r'.\dataset\MNIST\train-images-idx3-ubyte.gz'
    images, labels = load_mnist(train_images_path, train_labels_path, flatten=False)

    test_sets = {
        'original': images,
        'gaussian_noise': gaussian_noise(images, sigma=0.15),
        'small_rotation': rotate_images(images, angle=10),
        'scale_up': scale_images(images, scale=1.10),
        'scale_down': scale_images(images, scale=0.90),
    }

    summary = []
    cnn_preds_for_rotation = None
    rotated_180 = rotate_images(images, angle=180)

    for test_name, test_images in test_sets.items():
        for model_name, model in [('MLP', mlp), ('CNN', cnn)]:
            acc, preds, _ = evaluate_on_images(
                model,
                test_images,
                labels,
                name=f'{model_name}_{test_name}',
                save_confusion=(model_name == 'CNN'),
            )
            summary.append((model_name, test_name, acc))
            if model_name == 'CNN' and test_name == 'small_rotation':
                cnn_preds_for_rotation = preds

    acc, preds, _ = evaluate_on_images(
        cnn,
        rotated_180,
        labels,
        name='CNN_rotation_180_reserved_digits',
        save_confusion=False,
    )
    summary.append(('CNN', 'rotation_180_reserved_digits', acc))
    plot_reserved_digit_confusion_matrices(
        labels,
        preds,
        digit_groups=[(2, 5), (6, 9), (1, 0)],
    )

    save_summary(summary)
    return summary, cnn_preds_for_rotation


def save_summary(summary):
    save_path = os.path.join(EXTRA_RESULTS_DIR, 'robustness_summary.csv')
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('model,test,accuracy\n')
        for model_name, test_name, acc in summary:
            f.write(f'{model_name},{test_name},{acc:.6f}\n')

    print(f'Robustness summary saved to {save_path}')


def main():
    mlp, cnn = load_saved_models()

    visualize_saved_models()
    visualize_cnn_weights()
    robustness_tests(mlp, cnn)


if __name__ == '__main__':
    main()
