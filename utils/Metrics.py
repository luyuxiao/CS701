import numpy as np
import torch.nn as nn
import torch


def f1_score(y_hat, y_true, THRESHOLD=0.5):
    '''
    y_hat是未经过sigmoid函数激活的
    输出的f1为Marco-F1
    '''


    epsilon = 1e-7
    y_hat = y_hat > THRESHOLD
    y_hat = np.int8(y_hat)
    tp = np.sum(y_hat * y_true, axis=1)
    fp = np.sum(y_hat * (1 - y_true), axis=1)
    fn = np.sum((1 - y_hat) * y_true, axis=1)

    p = tp / (tp + fp + epsilon)  # epsilon的意义在于防止分母为0，否则当分母为0时python会报错
    r = tp / (tp + fn + epsilon)

    f1 = 2 * p * r / (p + r + epsilon)
    f1 = np.where(np.isnan(f1), np.zeros_like(f1), f1)
    return f1, p, r


class f1_loss(nn.Module):
    def __init__(self, epsilon=1e-15):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y_hat, y_true):
        # assert y_pred.ndim == 2
        # assert y_true.ndim == 1
        # y_true = F.one_hot(y_true, 2).to(torch.float32)
        # y_pred = F.softmax(y_pred, dim=1)

        tp = (y_hat * y_true).sum(dim=1)
        tn = (y_hat * y_true).sum(dim=1)
        fp = (y_hat * (1 - y_true)).sum(dim=1)
        fn = ((1 - y_hat) * y_true).sum(dim=1)

        soft_f1_class1 = 2 * tp / (2 * tp + fn + fp + self.epsilon)
        soft_f1_class0 = 2 * tn / (2 * tn + fn + fp + self.epsilon)

        cost_class1 = 1 - soft_f1_class1
        cost_class0 = 1 - soft_f1_class0

        cost = cost_class1 + cost_class0

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)
        #
        # f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        # f1 = f1.clamp(min=self.epsilon, max=1 - self.epsilon)
        return cost.mean()


class mixed_loss(nn.Module):
    def __init__(self, reduction, alpha, epsilon=1e-15):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
        self.alpha = alpha
        self.f1_loss = f1_loss()
        self.BCE_loss = nn.BCEWithLogitsLoss(reduction=self.reduction)

    def forward(self, y_hat, y_true):
        cost = self.alpha[0] * self.f1_loss(y_hat, y_true) + self.alpha[1] * self.BCE_loss(y_hat, y_true)
        return cost


if __name__ == "__main__":
    y_true = np.array([[1, 1, 0, 0, 1], [1, 0, 1, 1, 0], [0, 1, 1, 0, 0]])
    y_hat = np.array([[0, 1, 1, 1, 1], [1, 0, 0, 1, 1], [1, 0, 1, 0, 0]])

    f1 = f1_score(y_hat, y_true)
    print('F1 score:', f1)
