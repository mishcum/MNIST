import numpy as np
import torch
from torch import nn
from torch.utils import data
from torchvision.transforms import v2
from device import device

from MNISTDataset import MNISTDataset

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class Evaluation():
    def __init__(self, model : nn.Module, test_dataset):
        self.model = model

        dl_test = data.DataLoader(dataset=test_dataset, batch_size=len(test_dataset))
        X_test, y_test = next(iter(dl_test))
        X_test = X_test.to(device)

        with torch.no_grad():
            self.labels = list(test_dataset.format.keys())
            self.y_test = torch.argmax(y_test, dim=1).cpu().detach().numpy()
            self.predicts_proba = torch.softmax(self.model(X_test), dim=1).cpu().detach().numpy()
            self.predicts = np.argmax(self.predicts_proba, axis=1)

    def accuracy(self):
        acc = accuracy_score(self.y_test, self.predicts)
        print(f'Accuracy score: {acc:.2f}')
        return acc
    
    def precision_recall(self):
        prec = precision_score(self.y_test, self.predicts, average='macro')
        recall = recall_score(self.y_test, self.predicts, average='macro')
        print(f'Precision score: {prec:.2f}')
        print(f'Recall score: {prec:.2f}')
        return prec, recall
    
    def f1(self):
        f = f1_score(self.y_test, self.predicts, average='macro')
        print(f'F1 score: {f:.2f}')
        return f

