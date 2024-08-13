import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class OrbitalDataloader(Dataset):
    def __init__(self, data_folder, csv_file, task, transform=None):
        self.data_folder = data_folder
        self.csv_file = csv_file
        self.transform = transform
        self.task = task
        self.images = []
        self.molecule_data = {}
        self._load_data()

    def _load_data(self):
        """Load data from the CSV file and organize it into a dictionary."""
        data_df = pd.read_csv(self.csv_file)
        for index, row in data_df.iterrows():
            molecule_name = row['molecule_name']
            self.molecule_data[molecule_name] = {
                'molecule_energy': row['total_energy'],
                'mass_number': row['mass_number'],
                'atom_number': row['atom_number'],
                'neutron_number': row['neutron_number']
            }

        if self.task == 'single_element' or self.task == 'molecule':
            images = os.listdir(self.data_folder)
            for image in images:
                self.images.append(os.path.join(self.data_folder, image))
        else:  # self.task == 'unseen'
            folders = os.listdir(self.data_folder)
            for folder in folders:
                if os.path.isdir(os.path.join(self.data_folder, folder)):
                    images = os.listdir(os.path.join(self.data_folder, folder))
                    for image in images:
                        self.images.append(os.path.join(self.data_folder, folder, image))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path = self.images[idx]
        if self.task == 'single_element' or self.task == 'molecule':
            molecule_name = os.path.basename(image_path).split('_')[0]
        else:  # self.task == 'unseen'
            molecule_name = os.path.basename(os.path.dirname(image_path))

        data = self.molecule_data[molecule_name]
        molecule_energy = data['molecule_energy']
        mass_number = data['mass_number']
        atom_number = data['atom_number']
        neutron_number = data['neutron_number']

        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, molecule_energy, molecule_name, mass_number, atom_number, neutron_number


# Example usage:
# data_folder = '/path/to/data_folder'
# csv_file = '/path/to/csv_file.csv'
# task = 'single_element'  # or 'molecule' or 'unseen'
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
# dataset = MoleculeDataloader(data_folder, csv_file, task, transform)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
