import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from dataloader import OrbitalDataloader
from model import QuantumShellNet

# Set the criterion to L1 Loss
criterion = nn.L1Loss()

def test(model, test_loader, criterion, device):
    """Evaluate the model on the test set."""
    model.eval()
    running_loss = 0.0
    output = torch.zeros(0, dtype=torch.float32).to(device)

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            images, molecule_energies, molecule_names, mass_numbers, atom_numbers, neutron_numbers = data
            inputs = images.to(device)
            labels = molecule_energies.to(device)

            outputs = model(inputs, mass_numbers.to(device), atom_numbers.to(device), neutron_numbers.to(device))
            outputs = outputs.view(-1)
            loss = criterion(outputs, labels.float())
            running_loss += loss.item()
            output = torch.cat((output, outputs), 0)

        output = torch.mean(output)
        labels = labels[0]
    return running_loss / len(test_loader), output.to('cpu'), labels

def main():
    parser = argparse.ArgumentParser(description="Test the AI model for molecule property prediction.")
    parser.add_argument('--task', default='unseen',type=str, choices=['single_element', 'molecule', 'unseen'], required=True, help='Task type: single_element, molecule, or unseen.')
    parser.add_argument('--data_folder', default='test_data', type=str, required=True, help='Path to the data folder.')
    parser.add_argument('--csv_file', default='molecule_info.csv',type=str, required=True, help='Path to the CSV file.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for data loaders.')
    parser.add_argument('--load_folder', default='results',type=str, required=True, help='Root directory for saved models.')
    parser.add_argument('--seed_start', type=int, default=5, help='Start seed value.')
    parser.add_argument('--seed_end', type=int, default=26, help='End seed value.')
    parser.add_argument('--seed_step', type=int, default=5, help='Step size for seed values.')

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    custom_transforms = transforms.Compose([transforms.ToTensor()])

    test_dataset = OrbitalDataloader(f"{args.data_folder}", args.csv_file, args.task, transform=custom_transforms)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed_start))

    model_vs_avg_test_loss = {}

    for seed in range(args.seed_start, args.seed_end, args.seed_step):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

        model = QuantumShellNet()
        chkpt_path = f'{args.root}/unseen/{args.task}_{seed}/model.pth'
        model.load_state_dict(torch.load(chkpt_path))
        model.to(device)

        test_loss_list = []
        test_loss, output, labels = test(model, test_loader, criterion, device)
        test_loss_list.append(test_loss)
        avg_test_loss = np.mean(test_loss_list)
        print(f"Average test loss for seed {seed}: {avg_test_loss}")

        if args.task not in model_vs_avg_test_loss:
            model_vs_avg_test_loss[args.task] = []

        model_vs_avg_test_loss[args.task].append(avg_test_loss)

    for task in model_vs_avg_test_loss:
        avg_loss = np.mean(model_vs_avg_test_loss[task])
        std_loss = np.std(model_vs_avg_test_loss[task])
        model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Task: {task} - Avg Loss: {avg_loss}, Std Loss: {std_loss}, Model Params: {model_params}")

if __name__ == "__main__":
    main()
