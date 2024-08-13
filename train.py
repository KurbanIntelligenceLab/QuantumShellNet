import time

import argparse
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms
from tqdm import tqdm

from dataloader import OrbitalDataloader
from model import QuantumShellNet


def reset_weights(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


def train(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for i, data in enumerate(train_loader):
        images, molecule_energies, molecule_names, mass_numbers, atom_numbers, neutron_numbers = data
        inputs = images.to(device)
        labels = molecule_energies.to(device)

        optimizer.zero_grad()

        outputs = model(inputs, mass_numbers.to(device), atom_numbers.to(device), neutron_numbers.to(device))
        outputs = outputs.view(-1)
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            images, molecule_energies, molecule_names, mass_numbers, atom_numbers, neutron_numbers = data
            inputs = images.to(device)
            labels = molecule_energies.to(device)

            outputs = model(inputs, mass_numbers.to(device), atom_numbers.to(device), neutron_numbers.to(device))
            outputs = outputs.view(-1)
            loss = criterion(outputs, labels.float())

            running_loss += loss.item()

    return running_loss / len(val_loader)


def main():
    parser = argparse.ArgumentParser(description="Train and validate the AI model for molecule property prediction.")
    parser.add_argument('--task', default='unseen', type=str, choices=['single_element', 'molecule', 'unseen'], required=True, help='Task type: single_element, molecule, or unseen.')
    parser.add_argument('--data_folder', default='data', type=str, required=True, help='Path to the data folder.')
    parser.add_argument('--csv_file',default='molecule_info.csv', type=str, required=True, help='Path to the CSV file.')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate for the optimizer.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for data loaders.')
    parser.add_argument('--save_folder', default='results', type=str, required=True, help='Root directory for saving results.')
    parser.add_argument('--seed_start', type=int, default=5, help='Start seed value.')
    parser.add_argument('--seed_end', type=int, default=26, help='End seed value.')
    parser.add_argument('--seed_step', type=int, default=5, help='Step size for seed values.')

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.task == 'single_element':
        items = ['h', 'he', 'li', 'be', 'b', 'c', 'n', 'o', 'f', 'ne']
    elif args.task == 'molecule':
        items = ['li_h', 'li_li', 'n_n', 'c_o']
    else:
        items = ['unseen']

    for item in items:
        for seed in range(args.seed_start, args.seed_end, args.seed_step):
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)

            custom_transforms = transforms.Compose([transforms.ToTensor()])

            dataset = OrbitalDataloader(args.data_folder, args.csv_file, args.task, transform=custom_transforms)

            val_split = 0.2
            num_train = len(dataset)
            indices = list(range(num_train))
            split = int(np.floor(val_split * num_train))
            np.random.shuffle(indices)
            train_idx, val_idx = indices[split:], indices[:split]

            train_sampler = SubsetRandomSampler(train_idx, generator=torch.Generator().manual_seed(seed))
            val_sampler = SubsetRandomSampler(val_idx, generator=torch.Generator().manual_seed(seed))

            train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler, generator=torch.Generator().manual_seed(seed))
            val_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=val_sampler, generator=torch.Generator().manual_seed(seed))

            model = QuantumShellNet()
            model.to(device)
            reset_weights(model)

            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-3)

            train_loss = []
            val_loss = []
            fold_duration = []
            fold_start_time = time.time()

            for epoch in tqdm(range(args.num_epochs)):
                print(f"Epoch {epoch + 1} of {args.num_epochs}")
                train_epoch_loss = train(model, train_loader, optimizer, criterion, device)
                val_epoch_loss = validate(model, val_loader, criterion, device)
                train_loss.append(train_epoch_loss)
                val_loss.append(val_epoch_loss)
                print(f"Train Loss: {train_epoch_loss:.4f}, Validation Loss: {val_epoch_loss:.4f}")
                print(f"Train RMSE: {np.sqrt(train_epoch_loss):.4f}, Validation RMSE: {np.sqrt(val_epoch_loss):.4f}")

                task_seed_path = f"{item}/{args.task}_{seed}"
                if not os.path.exists(f'{args.root}/{task_seed_path}'):
                    os.makedirs(f'{args.root}/{task_seed_path}')

                np.savetxt(f'{args.root}/{task_seed_path}/train_loss.txt', train_loss)
                np.savetxt(f'{args.root}/{task_seed_path}/val_loss.txt', val_loss)

                if val_epoch_loss <= min(val_loss):
                    torch.save(model.state_dict(), f"{args.root}/{task_seed_path}/model.pth")
                    print("Saved best model weights!")

            fold_end_time = time.time()
            fold_duration.append(fold_end_time - fold_start_time)
            np.savetxt(f'{args.root}/{task_seed_path}/train_duration.txt', fold_duration)


if __name__ == "__main__":
    main()
