from dataset import PetDataset
from transforms import Rescale, RandomCrop, Normalize, ToTensor
from torchvision import models, transforms
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim import lr_scheduler
import torch.optim as optim
import torch.nn as nn
import numpy as np
import time
import copy

class Model:
    """
    Loads a pretrained resnet34 model and trains it
    """

    def __init__(self, model, dataset, device) -> None:
        self.dataset = dataset
        self.device = device
        self.model = model.to(device)

        self.random_seed = 42
        self.batch_size = 16

        self.dataloaders = {}
        self.dataset_sizes = {}
        self.train_test_split()

        # self.clean_data()

    def clean_data(self) -> None:
        print('cleaning data')
        print('length', len(self.dataset))
        for phase in ['train', 'val']:
            for inputs, labels in self.dataloaders[phase]:
                inputs = inputs.to(device=device, dtype=torch.float)
                labels = labels.to(device)

        print('resplitting data after cleaning')
        self.train_test_split()

    def train_test_split(self, validation_split: float = 0.2, shuffle_dataset: bool = True) -> None:
        # Create data indices for training and validation splits
        dataset_size = len(self.dataset)
        indices = list(range(dataset_size))
        split_idx = int(np.floor(validation_split * dataset_size))
        if shuffle_dataset:
            np.random.seed(self.random_seed)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split_idx:], indices[:split_idx]

        # Create data samplers and loaders
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)

        train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, sampler=train_sampler, num_workers=4)
        val_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, sampler=val_sampler, num_workers=4)

        self.dataloaders['train'] = train_loader
        self.dataloaders['val'] = val_loader
        self.dataset_sizes['train'] = len(train_indices)
        self.dataset_sizes['val'] = len(val_indices)

    def train_model(self, criterion, optimizer, scheduler, num_epochs: int) -> None:
        since = time.time()

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"Data length: {len(self.dataset)}")
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()   # Set model to validation mode

                running_loss = 0
                running_corrects = 0

                # Iterate over the data
                for inputs, labels in self.dataloaders[phase]:
                    inputs = inputs.to(device=device, dtype=torch.float)
                    labels = labels.to(device)

                    # Zero the parameter gradients
                    optimizer.zero_grad()

                    # Forward
                    # Only track history if in training
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # Backpropagation only in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # Statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects.double() / self.dataset_sizes[phase]

                print(f"Phase: {phase} Loss: {epoch_loss} Accuracy: {epoch_acc}")

                # Deep copy the best model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())

            print()

        time_elapsed = time.time() - since
        print(f"Training complete in {time_elapsed // 60} min, {time_elapsed % 60} sec")
        print(f"Best validation accuracy: {best_acc}")

        # Load best model weights
        self.model.load_state_dict(best_model_wts)
        return self.model
    

if __name__ == "__main__":
    transform = transforms.Compose([Rescale(256), RandomCrop(224), Normalize()])
    pet_data = PetDataset(transform=transform)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = models.resnet34(pretrained=True)

    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, len(pet_data.breed_to_int))

    criterion = nn.CrossEntropyLoss()

    # All parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    # Define and train the model
    model = Model(model_ft, pet_data, device)
    model.train_model(criterion, optimizer_ft, exp_lr_scheduler, num_epochs = 4)