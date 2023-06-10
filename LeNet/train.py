import os
import random

from PIL import Image

BATCH_SIZE = 256
NUM_EPOCHS = 20
LEARNING_RATE = 0.005
NUM_WORKERS = 4
CLASSES = 91

import torch
from torchvision import transforms
from torch.utils.data import Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AIDADataset(Dataset):
    def __init__(self, img_dir, limit, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.files = os.listdir(self.img_dir)
        print("dataset:", len(self.files))
        random.shuffle(self.files)
        self.files = self.files[:limit]
        self.default_transform = transforms.ToTensor()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        imname = self.files[idx]
        image = self.default_transform(Image.open(os.path.join(self.img_dir, imname)))
        label = int(imname.split("_")[0])
        assert 0 < label <= CLASSES, label
        label = torch.tensor(label)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class PyTorchLeNet5(torch.nn.Module):

    def __init__(self, num_classes, grayscale=False):
        super().__init__()

        self.grayscale = grayscale
        self.num_classes = num_classes

        if self.grayscale:
            in_channels = 1
        else:
            in_channels = 3

        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 6, kernel_size=5),
            torch.nn.Tanh(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(6, 16, kernel_size=5),
            torch.nn.Tanh(),
            torch.nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(16 * 5 * 5, 120),
            torch.nn.Tanh(),
            torch.nn.Linear(120, 84),
            torch.nn.Tanh(),
            torch.nn.Linear(84, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        logits = self.classifier(x)
        return logits


import pytorch_lightning as pl
import torchmetrics


# LightningModule that receives a PyTorch model as input
class LightningModel(pl.LightningModule):
    def __init__(self, model, learning_rate):
        super().__init__()

        self.learning_rate = learning_rate
        # The inherited PyTorch module
        self.model = model

        # Save settings and hyperparameters to the log directory
        # but skip the model parameters
        self.save_hyperparameters(ignore=['model'])

        # Set up attributes for computing the accuracy
        self.train_acc = torchmetrics.Accuracy('multiclass', num_classes=CLASSES)
        self.valid_acc = torchmetrics.Accuracy('multiclass', num_classes=CLASSES)
        self.test_acc = torchmetrics.Accuracy('multiclass', num_classes=CLASSES)

    # Defining the forward method is only necessary
    # if you want to use a Trainer's .predict() method (optional)
    def forward(self, x):
        return self.model(x)

    # A common forward step to compute the loss and labels
    # this is used for training, validation, and testing below
    def _shared_step(self, batch):
        features, true_labels = batch
        logits = self(features)
        loss = torch.nn.functional.cross_entropy(logits, true_labels)
        predicted_labels = torch.argmax(logits, dim=1)

        return loss, true_labels, predicted_labels

    def training_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.log("train_loss", loss)

        # To account for Dropout behavior during evaluation
        self.model.eval()
        with torch.no_grad():
            _, true_labels, predicted_labels = self._shared_step(batch)
        self.train_acc.update(predicted_labels, true_labels)
        self.log("train_acc", self.train_acc, on_epoch=True, on_step=False)
        self.model.train()
        return loss  # this is passed to the optimizer for training

    def validation_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.log("valid_loss", loss)
        self.valid_acc(predicted_labels, true_labels)
        self.log("valid_acc", self.valid_acc,
                 on_epoch=True, on_step=False, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.test_acc(predicted_labels, true_labels)
        self.log("test_acc", self.test_acc, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


from torch.utils.data.dataset import random_split, Dataset
from torch.utils.data import DataLoader


class DataModule(pl.LightningDataModule):
    def __init__(self, data_path='./', limit=-1):
        super().__init__()
        self.data_path = data_path
        self.limit = limit

    def prepare_data(self):
        self.resize_transform = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.Resize((32, 32))
            ]
        )

        return

    def setup(self, stage=None):
        # Note transforms.ToTensor() scales input images
        # to 0-1 range
        train = AIDADataset(self.data_path, self.limit, self.resize_transform)

        self.train, self.test, self.valid = random_split(train, [0.7, 0.2, 0.1])

    def train_dataloader(self):
        train_loader = DataLoader(dataset=self.train,
                                  batch_size=BATCH_SIZE,
                                  drop_last=True,
                                  shuffle=True,
                                  num_workers=NUM_WORKERS,
                                  pin_memory=True,
                                  pin_memory_device=str(device)
                                  )
        return train_loader

    def val_dataloader(self):
        valid_loader = DataLoader(dataset=self.valid,
                                  batch_size=BATCH_SIZE,
                                  drop_last=False,
                                  shuffle=False,
                                  num_workers=NUM_WORKERS,
                                  pin_memory=True,
                                  pin_memory_device=str(device)
                                  )
        return valid_loader

    def test_dataloader(self):
        test_loader = DataLoader(dataset=self.test,
                                 batch_size=BATCH_SIZE,
                                 drop_last=False,
                                 shuffle=False,
                                 num_workers=NUM_WORKERS,
                                 pin_memory=True,
                                 pin_memory_device=str(device))
        return test_loader


def main():
    torch.manual_seed(1)
    data_module = DataModule(data_path='../dataset/AIDA2', limit=-1)
    data_module.prepare_data()
    data_module.setup()

    # Checking the dataset
    all_train_labels = []
    all_test_labels = []

    for images, labels in data_module.train:  # todo
        all_train_labels.append(labels)

    all_train_labels = torch.tensor(all_train_labels)

    for images, labels in data_module.test:  # todo
        all_test_labels.append(labels)
    all_test_labels = torch.tensor(all_test_labels)

    print('Training labels:', torch.unique(all_train_labels))
    print('Training label distribution:', torch.bincount(all_train_labels))

    print('\nTest labels:', torch.unique(all_test_labels))
    print('Test label distribution:', torch.bincount(all_test_labels))

    majority_prediction = torch.argmax(torch.bincount(all_test_labels))
    baseline_acc = torch.mean((all_test_labels == majority_prediction).float())
    print(f'Baseline ACC: {baseline_acc * 100:.2f}%')

    import os

    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger

    pytorch_model = PyTorchLeNet5(
        num_classes=CLASSES, grayscale=True)

    lightning_model = LightningModel(
        model=pytorch_model, learning_rate=LEARNING_RATE)

    lightning_model.to(device)

    import time

    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS,
        accelerator="auto",  # Uses GPUs or TPUs if available
        devices="auto",  # Uses all available GPUs/TPUs if applicable
    )

    start_time = time.time()
    trainer.fit(model=lightning_model, datamodule=data_module, ckpt_path="last")

    runtime = (time.time() - start_time) / 60
    print(f"Training took {runtime:.2f} min in total.")

    trainer.test(model=lightning_model, datamodule=data_module)

    lightning_model.eval()

    trainer.save_checkpoint("checkpoint/res.chkpt")

    test_dataloader = data_module.test_dataloader()

    all_true_labels = []
    all_predicted_labels = []
    for batch in test_dataloader:
        features, labels = batch

        with torch.no_grad():  # since we don't need to backprop
            logits = lightning_model(features)
        predicted_labels = torch.argmax(logits, dim=1)
        all_predicted_labels.append(predicted_labels)
        all_true_labels.append(labels)

    all_predicted_labels = torch.cat(all_predicted_labels)
    all_true_labels = torch.cat(all_true_labels)
    print(all_predicted_labels[:5])

    test_acc = torch.mean((all_predicted_labels == all_true_labels).float())
    print(f'Test accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)')

    img, lab = data_module.test[100]

    image_nchw = img.unsqueeze(0)
    print(image_nchw.shape)
    st = time.time()

    with torch.no_grad():  # since we don't need to backprop
        logits = lightning_model(image_nchw)
        probas = torch.softmax(logits, dim=1)
        predicted_label = torch.argmax(probas)

    print(time.time() - st)
    print(f'Predicted label: {predicted_label}')
    print(f'Actual label: {lab}')
    print(f'Class-membership probability {probas[0][predicted_label] * 100:.2f}%')


if __name__ == '__main__':
    main()
