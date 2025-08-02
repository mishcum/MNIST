import torch
from torch.utils import data
from model import ModelCNN
from MNISTDataset import MNISTDataset
import torchvision.transforms.v2 as v2

from train import Trainer
from evaluate import Evaluation

if __name__ == '__main__':
    model = ModelCNN()
    train_dataset = MNISTDataset(path='data', train=True, transforms=v2.Compose([v2.ToImage(), v2.Grayscale(), v2.ToDtype(dtype=torch.float32), v2.Normalize((0.1307,), (0.3081,))]))

    dataset_val, dataset_train = data.random_split(dataset=train_dataset, lengths=[0.3, 0.7])
    dl_train = data.DataLoader(dataset=dataset_train, batch_size=32, shuffle=True)
    dl_val = data.DataLoader(dataset=dataset_val, batch_size=len(dataset_val))
    
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001, weight_decay=0.0001)

    trainer = Trainer(model, epochs=20, loss_func=loss_func, optimizer=optimizer, train_set=dl_train, val_set=dl_val)
    model = trainer.train()
    trainer.train_plot('CNN model training', save=True)

    test_dataset = MNISTDataset(path='data', train=False, transforms=v2.Compose([v2.ToImage(), v2.Grayscale(), v2.ToDtype(dtype=torch.float32), v2.Normalize((0.1307,), (0.3081,))]))

    evaluation = Evaluation(model, test_dataset)
    evaluation.accuracy()
    evaluation.precision_recall()
    evaluation.f1()