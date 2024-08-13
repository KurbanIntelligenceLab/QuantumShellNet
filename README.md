# QuantumShellNet
Official repo for the QuantumShellNet

This repository contains a unified training script for predicting molecule properties using a deep learning model. The script supports training for single elements, molecules, and unseen elements using the `QuantumShellNet` model.

## Requirements

- Python 3.10+
- PyTorch
- torchvision
- numpy
- pandas
- Pillow
- tqdm
- matplotlib

## Setup

1. Clone the repository:

    ```bash
    git clone git@github.com:KurbanIntelligenceLab/QuantumShellNet.git
    cd QuantumShellNet
    ```

2. Install the required packages:
    Install PyTorch via official webpage (https://pytorch.org/get-started/locally/). For the rest use the requirements.txt file
    ```bash
    pip install -r requirements.txt
    ```
## Data

You can download the data for elements and molecules from the following link: [QuantumShellNet Data](https://tamucs-my.sharepoint.com/:f:/r/personal/hasan_kurban_tamu_edu/Documents/KIL-OneDrive/Can%20Polat/QuantumShellNet/data?csf=1&web=1&e=fKnD5n)

## Train

The training script can be executed with various parameters. Below are the available arguments:

- `--task`: Task type (`single_element`, `molecule`, or `unseen`) (required).
- `--data_folder`: Path to the data folder (required).
- `--csv_file`: Path to the CSV file (required).
- `--num_epochs`: Number of training epochs (default: 100).
- `--learning_rate`: Learning rate for the optimizer (default: 0.01).
- `--batch_size`: Batch size for data loaders (default: 1).
- `--root`: Root directory for saving results (required).
- `--seed_start`: Start seed value (default: 5).
- `--seed_end`: End seed value (default: 26).
- `--seed_step`: Step size for seed values (default: 5).

### Example Commands

#### For Single Elements:
```bash
python train.py --task single_element --data_folder "/path/to/data" --csv_file "/path/to/csv" --save_folder "/path/to/root"
```

#### For Molecules:
```bash
python train.py --task molecule --data_folder "/path/to/data" --csv_file "/path/to/csv" --save_folder "/path/to/root"
```

#### For Unseen Elements:
```bash
python train.py --task unseen --data_folder "/path/to/data" --csv_file "/path/to/csv" --save_folder "/path/to/root"
```

### Script Details

#### `train.py`

This script trains a model to predict various physical properties of molecules. It supports training for single elements, molecules, and unseen elements. The data is split into training and validation sets, and the best model weights are saved based on the validation loss.

##### Arguments

- `--task`: Task type (`single_element`, `molecule`, or `unseen`) (required).
- `--data_folder`: Path to the data folder (required).
- `--csv_file`: Path to the CSV file (required).
- `--num_epochs`: Number of epochs for training (default: 100).
- `--learning_rate`: Learning rate for the optimizer (default: 0.01).
- `--batch_size`: Batch size for data loading (default: 1).
- `--root`: Root directory for saving results (required).
- `--seed_start`: Start seed value (default: 5).
- `--seed_end`: End seed value (default: 26).
- `--seed_step`: Step size for seed values (default: 5).

##### Functions

- `reset_weights(model)`: Resets the weights of the model.
- `train(model, train_loader, optimizer, criterion, device)`: Trains the model for one epoch.
- `validate(model, val_loader, criterion, device)`: Validates the model.

### Results

The training and validation losses are saved in the specified `root` directory, along with the model weights of the best performing model based on the validation loss. 

Example directory structure for saved results:
```
results/
├── single_element/
│   ├── atom/
│   │   ├── task_seed/
│   │   │   ├── train_loss.txt
│   │   │   ├── val_loss.txt
│   │   │   ├── model.pth
│   │   │   ├── train_duration.txt
...
```

## Test

The test script can be executed with various parameters. Below are the available arguments:

- `--task`: Task type (`single_element`, `molecule`, or `unseen`) (required).
- `--data_folder`: Path to the data folder (required).
- `--csv_file`: Path to the CSV file (required).
- `--batch_size`: Batch size for data loaders (default: 64).
- `--root`: Root directory for saved models (required).
- `--seed_start`: Start seed value (default: 5).
- `--seed_end`: End seed value (default: 26).
- `--seed_step`: Step size for seed values (default: 5).

### Example Commands

#### For Single Elements:
```bash
python test.py --task single_element --data_folder "/path/to/data" --csv_file "/path/to/csv" --root "/path/to/root"
```

#### For Molecules:
```bash
python test.py --task molecule --data_folder "/path/to/data" --csv_file "/path/to/csv" --root "/path/to/root"
```

#### For Unseen Elements:
```bash
python test.py --task unseen --data_folder "/path/to/data" --csv_file "/path/to/csv" --root "/path/to/root"
```

### Script Details

#### `test.py`

This script evaluates a trained model to predict various physical properties of molecules. It supports testing for single elements, molecules, and unseen elements. The script calculates the average test loss across different seeds and reports the mean and standard deviation of the losses.

##### Arguments

- `--task`: Task type (`single_element`, `molecule`, or `unseen`) (required).
- `--data_folder`: Path to the data folder (required).
- `--csv_file`: Path to the CSV file (required).
- `--batch_size`: Batch size for data loading (default: 64).
- `--root`: Root directory for saved models (required).
- `--seed_start`: Start seed value (default: 5).
- `--seed_end`: End seed value (default: 26).
- `--seed_step`: Step size for seed values (default: 5).

### Results

The script prints the average and standard deviation of the test losses for each task, along with the number of model parameters. 

Example output:

```
Task: single_element - Avg Loss: 0.1234, Std Loss: 0.0123, Model Params: 123456
Task: molecule - Avg Loss: 0.2345, Std Loss: 0.0234, Model Params: 123456
Task: unseen - Avg Loss: 0.3456, Std Loss: 0.0345, Model Params: 123456
```
