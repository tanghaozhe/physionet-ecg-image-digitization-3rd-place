import numpy as np
import torch

def np_snr(predict, truth):
    eps = 1e-7
    signal = (truth ** 2).sum()
    noise = ((predict - truth) ** 2).sum()
    snr = signal / (noise + eps)
    snr_db = 10 * np.log10(snr + eps)
    return snr_db

def torch_snr(predict, truth):
    eps = 1e-7
    signal = (truth ** 2).sum()
    noise = ((predict - truth) ** 2).sum()
    snr = signal / (noise + eps)
    snr_db = 10 * torch.log10(snr + eps)
    return snr_db

def compute_iou(pred, target, threshold=0.5):
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()

    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum() - intersection

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    iou = intersection / union
    return iou.item()

def compute_dice(pred, target, threshold=0.5):
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()

    intersection = (pred_binary * target_binary).sum()
    total = pred_binary.sum() + target_binary.sum()

    if total == 0:
        return 1.0 if intersection == 0 else 0.0

    dice = (2.0 * intersection) / total
    return dice.item()

def compute_pixel_accuracy(pred, target, threshold=0.5):
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()

    correct = (pred_binary == target_binary).sum()
    total = target_binary.numel()

    return (correct / total).item()

def compute_precision_recall(pred, target, threshold=0.5):
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()

    tp = (pred_binary * target_binary).sum()
    fp = (pred_binary * (1 - target_binary)).sum()
    fn = ((1 - pred_binary) * target_binary).sum()

    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)

    return precision.item(), recall.item()

class SegmentationMetrics:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.iou_sum = 0.0
        self.dice_sum = 0.0
        self.accuracy_sum = 0.0
        self.precision_sum = 0.0
        self.recall_sum = 0.0
        self.count = 0

    def update(self, pred, target):
        if pred.dim() == 4:
            B = pred.size(0)
            for i in range(B):
                iou = compute_iou(pred[i], target[i], self.threshold)
                dice = compute_dice(pred[i], target[i], self.threshold)
                accuracy = compute_pixel_accuracy(pred[i], target[i], self.threshold)
                precision, recall = compute_precision_recall(pred[i], target[i], self.threshold)

                self.iou_sum += iou
                self.dice_sum += dice
                self.accuracy_sum += accuracy
                self.precision_sum += precision
                self.recall_sum += recall
                self.count += 1
        else:
            iou = compute_iou(pred, target, self.threshold)
            dice = compute_dice(pred, target, self.threshold)
            accuracy = compute_pixel_accuracy(pred, target, self.threshold)
            precision, recall = compute_precision_recall(pred, target, self.threshold)

            self.iou_sum += iou
            self.dice_sum += dice
            self.accuracy_sum += accuracy
            self.precision_sum += precision
            self.recall_sum += recall
            self.count += 1

    def get_metrics(self):
        if self.count == 0:
            return {
                'iou': 0.0,
                'dice': 0.0,
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0
            }

        precision = self.precision_sum / self.count
        recall = self.recall_sum / self.count
        f1 = 2 * precision * recall / (precision + recall + 1e-7)

        return {
            'iou': self.iou_sum / self.count,
            'dice': self.dice_sum / self.count,
            'accuracy': self.accuracy_sum / self.count,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
