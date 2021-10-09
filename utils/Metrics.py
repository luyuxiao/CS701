import numpy as np


def f1_loss(y_hat, y_true, THRESHOLD=0.5):
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
    return f1


if __name__ == "__main__":
    y_true = np.array([[1, 1, 0, 0, 1], [1, 0, 1, 1, 0], [0, 1, 1, 0, 0]])
    y_hat = np.array([[0, 1, 1, 1, 1], [1, 0, 0, 1, 1], [1, 0, 1, 0, 0]])

    f1 = f1_loss(y_hat, y_true)
    print('F1 score:', f1)
