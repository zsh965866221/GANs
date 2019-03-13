# coding = utf-8
# 评估方法：统计每一类的IOU
# 使用numpy


import numpy as np
from collections import defaultdict


class IOU:
    """
    统计每一类分对的像素，记录三个信息：[每一类真值的pix个数，每一类预测的pix个数，预测正确的pix个数]
    不应该包含background，VOC一共有20类, 不包括0[1, 20]
    """
    def __init__(self, n_class=21):
        self.n_class = n_class
        self.trues = defaultdict(list)
        self.predictions = defaultdict(list)
        self.hitting = defaultdict(list)
        self.eps = 1e-8

    def clear(self):
        self.trues.clear()
        self.predictions.clear()
        self.hitting.clear()

    def update(self, targets, predictions):
        for index in range(1, self.n_class):
            i_t = (targets == index)
            i_p = (predictions == index)
            self.trues[index].append(np.sum(i_t))
            self.predictions[index].append(np.sum(i_p))
            self.hitting[index].append(np.sum(np.logical_and(i_t, i_p)))

    def compute(self):
        r = defaultdict(float)
        for index in range(1, self.n_class):
            i = np.sum(self.hitting[index])
            t = np.sum(self.trues[index])
            p = np.sum(self.predictions[index])
            r[index] = float(i) / (float(t) + float(p) - float(i) + self.eps)
        return r

