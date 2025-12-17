import os
import json
import torch
from torch import nn
import argparse
import pandas as pd
import numpy as np
from src.utils.basic_funcs import set_seed
from src.analysis import ann as ann
from torch.utils.data import Dataset, DataLoader
from src.utils import basic_funcs as basic
from src.models.network_interface import InterferenceTaskNetwork
from src.models.rnn_networks import SimpleRNN
import math
import copy
from tqdm.auto import tqdm
from scipy import stats
import math


"""
HELPER FUNCTIONS
"""
def numpy_to_python(obj):
    """Convert numpy objects to Python native types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [numpy_to_python(item) for item in obj]
    return obj
    
def filter_participant_data(df, participant, task_section):
    """
    Filter participant data by task section.
    """
    return df.loc[
        (df['participant'] == participant) & (df['task_section'] == task_section),
        ['index', 'feature_idx', 'feat_val', 'noisy_feedback_value', 'stimID','test_trial']
    ].reset_index(drop=True)

def adjust_indices(participant_data, offset):
    """
    Adjust the indices of participant data by the specified offset.
    """
    participant_data['index'] -= offset
    return participant_data.reset_index(drop=True)

def create_inputs_matrix(participant_data, n_stim_per_task):
    """
    Create an inputs matrix with one-hot encoded stimulus IDs.
    """
    length = participant_data.shape[0]
    inputs = np.zeros((length, n_stim_per_task * 2))
    for index, row in participant_data.iterrows():
        inputs[index, int(row['stimID'])] = 1
    return inputs

def process_raw_inputs_and_labels(participant_data, n_stim_per_task, task_idx):
    """
    Process raw inputs and labels for a given task.
    """
    unique_inputs = participant_data['stimID'].unique().astype(int)
    raw_inputs = np.full((n_stim_per_task, n_stim_per_task * 2), np.nan, dtype=np.float32)
    raw_labels = np.full((4, n_stim_per_task), np.nan, dtype=np.float32)

    for idx, stim_id in enumerate(unique_inputs):
        feat1 = participant_data.loc[
            (participant_data['stimID'] == stim_id) & (participant_data['feature_idx'] == 0), 'feat_val'
        ].unique()
        feat2 = participant_data.loc[
            (participant_data['stimID'] == stim_id) & (participant_data['feature_idx'] == 1), 'feat_val'
        ].unique()
        raw_labels[0, idx] = np.cos(feat1)[0]
        raw_labels[1, idx] = np.sin(feat1)[0]
        raw_labels[2, idx] = np.cos(feat2)[0]
        raw_labels[3, idx] = np.sin(feat2)[0]

        input_skeleton = np.zeros((n_stim_per_task * 2))
        input_skeleton[stim_id] = 1
        raw_inputs[idx, :] = input_skeleton

    return raw_inputs, raw_labels

def assemble_dataset(participant_data, inputs, label_cos, label_sin):
    """
    Assemble the dataset dictionary for a task.
    """
    return {
        'index': participant_data['index'].values,
        'stim_index': participant_data['stimID'].values,
        'input': inputs,
        'feature_probe': participant_data['feature_idx'].values,
        'test_stim': participant_data['test_trial'].values,
        'label_x': label_cos,
        'label_y': label_sin,
    }
    
def get_datasets(df, participant, task_parameters):
    """
    Main function to get datasets and process raw inputs and labels.
    """
    # Filter data for each task section
    participant_training_A1 = filter_participant_data(df, participant, 'A1')
    participant_training_B = filter_participant_data(df, participant, 'B')
    participant_training_A2 = filter_participant_data(df, participant, 'A2')

    # Adjust indices for B and C
    A_length = len(participant_training_A1)
    B_length = len(participant_training_B)
    participant_training_B = adjust_indices(participant_training_B, A_length)
    participant_training_A2 = adjust_indices(participant_training_A2, A_length + B_length)

    # Create inputs matrices
    A1_inputs = create_inputs_matrix(participant_training_A1, task_parameters['nStim_perTask'])
    B_inputs = create_inputs_matrix(participant_training_B, task_parameters['nStim_perTask'])
    A2_inputs = create_inputs_matrix(participant_training_A2, task_parameters['nStim_perTask'])

    # Process raw inputs and labels
    raw_inputs = np.full((3, task_parameters['nStim_perTask'], task_parameters['nStim_perTask'] * 2), np.nan, dtype=np.float32)
    raw_labels = np.full((3, 4, task_parameters['nStim_perTask']), np.nan, dtype=np.float32)

    raw_inputs[0], raw_labels[0] = process_raw_inputs_and_labels(participant_training_A1, task_parameters['nStim_perTask'], 0)
    raw_inputs[1], raw_labels[1] = process_raw_inputs_and_labels(participant_training_B, task_parameters['nStim_perTask'], 1)
    raw_inputs[2], raw_labels[2] = process_raw_inputs_and_labels(participant_training_A2, task_parameters['nStim_perTask'], 2)

    # Assemble datasets
    dataset_A1 = assemble_dataset(participant_training_A1, A1_inputs, np.cos(participant_training_A1['feat_val'].values), np.sin(participant_training_A1['feat_val'].values))
    dataset_B = assemble_dataset(participant_training_B, B_inputs, np.cos(participant_training_B['feat_val'].values), np.sin(participant_training_B['feat_val'].values))
    dataset_A2 = assemble_dataset(participant_training_A2, A2_inputs, np.cos(participant_training_A2['feat_val'].values), np.sin(participant_training_A2['feat_val'].values))

    return dataset_A1, dataset_B, dataset_A2, raw_inputs, raw_labels

class CreateParticipantDataset(Dataset):
    """PyTorch Dataset for participant data."""
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset['index'])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = {key: self.dataset[key][idx] for key in self.dataset}
        if self.transform:
            sample = self.transform(sample)
        return sample

def compute_accuracy(predictions, ground_truth):
    """Compute accuracy between predictions and ground truth in radians."""
    predictions = np.asarray(predictions)
    ground_truth = np.asarray(ground_truth)
    wrapped_difference = basic.wrap_to_pi(predictions - ground_truth)
    normalized_error = np.abs(wrapped_difference) / np.pi
    return 1 - normalized_error

def batch_to_torch(numpy_version):
    """Convert numpy batch to torch tensor."""
    return numpy_version.type(torch.FloatTensor)

def train_participant_schedule(network, trainloader, n_epochs, loss_function, optimizer, do_update, do_test):
    """
    Train the network on x-y coordinates.
    
    This function now works with any network implementing InterferenceTaskNetwork interface.
    
    Args:
        network: Network instance implementing InterferenceTaskNetwork
        trainloader: DataLoader for training data
        n_epochs: Number of training epochs
        loss_function: Loss function (e.g., nn.MSELoss)
        optimizer: Optimizer (e.g., torch.optim.SGD)
        do_update: Update mode (0=no update, 1=standard, 2=conditional on feature_probe)
        do_test: Whether test trials are included (affects update logic)
    
    Returns:
        tuple: Various metrics including indexes, inputs, labels, probes, losses, accuracy, 
               predictions, hiddens, embeddings, readouts.
    """
    # Initialize storage lists
    metrics = {
        "indexes": [],
        "losses": [],
        "accuracy": [],
        "predictions": [],
        "hiddens": [],
        "embeddings": [],
        "readouts": [],
        "probes": [],
        "test_stim":[],
        "labels": [],
        "inputs": [],
    }

    # Reset hidden state at start of training (for RNNs)
    if network.supports_sequences:
        batch_size = trainloader.batch_size if hasattr(trainloader, 'batch_size') else 1
        network.reset_hidden_state(batch_size=batch_size)

    for epoch in range(n_epochs):
        # Optionally reset hidden state at start of each epoch (for RNNs)
        # Commented out to allow state to persist across epochs within a phase
        # if network.supports_sequences:
        #     batch_size = trainloader.batch_size if hasattr(trainloader, 'batch_size') else 1
        #     network.reset_hidden_state(batch_size=batch_size)
        for batch_idx, data in enumerate(trainloader):
            # Reset gradients
            optimizer.zero_grad()

            # Extract batch data
            index = data['stim_index']
            input = batch_to_torch(data['input'])
            label_x = batch_to_torch(data['label_x'])
            label_y = batch_to_torch(data['label_y'])
            feature_probe = batch_to_torch(data['feature_probe'])
            test_stim = batch_to_torch(data['test_stim'])
            
                    
            joined_label = torch.cat((label_x.unsqueeze(1), label_y.unsqueeze(1)), dim=1)
            radians_label = math.atan2(label_x, label_y)

            # Forward pass - use interface method
            out, hid = network(input)

            # Calculate loss based on feature probe
            # Handle both scalar and batch feature_probe
            if isinstance(feature_probe, torch.Tensor):
                feature_probe_val = feature_probe[0].item() if feature_probe.numel() > 0 else feature_probe.item()
            else:
                feature_probe_val = feature_probe
            
            if feature_probe_val == 0:
                loss = loss_function(out[:, :2], joined_label)
                pred_rads = math.atan2(out[:, 0].detach().cpu().numpy(), out[:, 1].detach().cpu().numpy())
                accuracy = compute_accuracy(pred_rads, radians_label.cpu().numpy())
                
            elif feature_probe_val == 1:
                loss = loss_function(out[:, 2:4], joined_label)
                pred_rads = math.atan2(out[:, 2].detach().cpu().numpy(), out[:, 3].detach().cpu().numpy())
                accuracy = compute_accuracy(pred_rads, radians_label.cpu().numpy())
                
            else:
                raise ValueError("Undefined loss setting for feature_probe.")

            # Update network if required
            if isinstance(test_stim, torch.Tensor):
                test_stim_val = test_stim[0].item() if test_stim.numel() > 0 else test_stim.item()
            else:
                test_stim_val = test_stim
                
            if do_update == 1 and do_test == 1 and test_stim_val == 0:
              loss.backward()
              optimizer.step()
            elif do_update == 1 and do_test == 0:
              loss.backward()
              optimizer.step()
            elif do_update == 2 and feature_probe_val == 0:  # In A2, only update for feature 0 
              loss.backward()
              optimizer.step()

            # Store metrics
            metrics["indexes"].append(index)
            metrics["inputs"].append(input.detach().cpu().numpy())
            metrics["labels"].append(joined_label.detach().cpu().numpy())
            metrics["probes"].append(feature_probe.detach().cpu().numpy() if isinstance(feature_probe, torch.Tensor) else feature_probe)
            metrics["test_stim"].append(test_stim.detach().cpu().numpy() if isinstance(test_stim, torch.Tensor) else test_stim)
            metrics["losses"].append(loss.item())
            metrics["accuracy"].append(accuracy)
            metrics["predictions"].append(np.expand_dims(out.detach().cpu().numpy(), axis=1))
            
            # Handle hidden state - may be multi-dimensional for RNNs
            hid_np = hid.detach().cpu().numpy()
            # If hidden is multi-dimensional (e.g., from RNN), take last layer or flatten appropriately
            if hid_np.ndim > 2:
                # For RNN: (num_layers, batch_size, hidden_dim) -> take last layer
                if hasattr(network, '_num_layers') and hid_np.shape[0] == network._num_layers:
                    hid_np = hid_np[-1]  # Take last layer
                else:
                    # Flatten or take mean across layers
                    hid_np = hid_np.reshape(hid_np.shape[0], -1)
            metrics["hiddens"].append(hid_np)
            
            # Get embeddings and readouts using interface methods (handle None)
            embeddings = network.get_embeddings()
            if embeddings is not None:
                metrics["embeddings"].append(embeddings.detach().cpu().numpy())
            else:
                # Store None or placeholder - will need to handle in array conversion
                metrics["embeddings"].append(None)
            
            readouts = network.get_readouts()
            if readouts is not None:
                metrics["readouts"].append(readouts.detach().cpu().numpy())
            else:
                metrics["readouts"].append(None)

    # Convert lists to arrays where applicable
    # Handle None values in embeddings/readouts
    for key in ["embeddings", "readouts"]:
        if metrics[key] and all(x is None for x in metrics[key]):
            # All None - create empty array or skip
            metrics[key] = np.array([])
        elif metrics[key] and any(x is None for x in metrics[key]):
            # Mixed None and arrays - filter out None or use placeholder
            metrics[key] = [x for x in metrics[key] if x is not None]
            if metrics[key]:
                metrics[key] = np.array(metrics[key])
            else:
                metrics[key] = np.array([])
        else:
            metrics[key] = np.squeeze(np.array(metrics[key])) if metrics[key] else np.array([])
    
    # Convert other metrics
    for key in ["indexes", "inputs", "labels", "probes", "test_stim", "losses", 
                "accuracy", "predictions", "hiddens"]:
        if metrics[key]:
            metrics[key] = np.squeeze(np.array(metrics[key]))
        else:
            metrics[key] = np.array([])
    
    return (
        metrics["indexes"],
        metrics["inputs"],
        metrics["labels"],
        metrics["probes"],
        metrics["test_stim"],
        metrics["losses"],
        metrics["accuracy"],
        metrics["predictions"],
        metrics["hiddens"],
        metrics["embeddings"],
        metrics["readouts"],
    )

class simpleLinearNet(InterferenceTaskNetwork):
    """A simple linear neural network with one hidden layer.
    
    Architecture:
    input -> hidden layer -> output
    All layers are fully connected with no bias terms.
    
    This class implements the InterferenceTaskNetwork interface for compatibility
    with the abstracted training pipeline.
    """
    def __init__(self, dim_input, dim_hidden, dim_output):
        super(simpleLinearNet, self).__init__()
        self._dim_input = dim_input
        self._dim_hidden = dim_hidden
        self._dim_output = dim_output
        self.in_hid = nn.Linear(dim_input, dim_hidden, bias=False)
        self.hid_out = nn.Linear(dim_hidden, dim_output, bias=False)
        
    def forward(self, x, hidden=None):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor (batch_size, input_dim)
            hidden: Ignored for FFN (kept for interface compatibility)
        
        Returns:
            output: Network output (batch_size, output_dim)
            hidden: Hidden layer activations (batch_size, dim_hidden)
        """
        hid = self.in_hid(x)
        out = self.hid_out(hid)
        return out, hid
    
    def get_hidden_state(self):
        """Get current hidden state. Returns None for FFNs."""
        return None
    
    def reset_hidden_state(self, batch_size=1):
        """Reset hidden state. No-op for FFNs."""
        pass
    
    def get_embeddings(self):
        """Get input embedding weights (input-to-hidden layer)."""
        return self.in_hid.weight
    
    def get_readouts(self):
        """Get output readout weights (hidden-to-output layer)."""
        return self.hid_out.weight
    
    @property
    def supports_sequences(self):
        """FFN does not support sequences."""
        return False
    
    @property
    def input_dim(self):
        """Input dimension."""
        return self._dim_input
    
    @property
    def output_dim(self):
        """Output dimension."""
        return self._dim_output

def ex_initializer_(model, gamma=1e-3, mean=0.0):
    """
    In-place Re-initialization of weights.
    
    This function now works generically with any network architecture by using
    named_parameters() to find all weight parameters.

    Args:
        model: torch.nn.Module
            PyTorch neural net model (should implement InterferenceTaskNetwork)
        
        gamma: float
            Initialization scale for hidden/input layers
        
        mean: float
            Mean for weight initialization

    Returns:
        Nothing
    """
    for name, param in model.named_parameters():
        if "weight" in name:
            # Handle different parameter shapes (2D for linear, 3D+ for conv, etc.)
            if param.dim() >= 2:
                # For 2D weights (Linear layers): (out_features, in_features)
                # For higher dims, use first two dimensions
                n_out, n_in = param.shape[0], param.shape[1]
                
                # Determine initialization std based on layer type
                # Check for output layer patterns (more flexible matching)
                if any(pattern in name.lower() for pattern in ["out", "output", "readout", "final", "classifier"]):
                    # Output layer weights - smaller initialization
                    std = 1e-3
                elif any(pattern in name.lower() for pattern in ["rnn", "lstm", "gru"]):
                    # RNN weights - use gamma but may need special handling
                    # For RNN, typically use smaller init for recurrent weights
                    if "hh" in name.lower():  # hidden-to-hidden
                        std = gamma * 0.1  # Smaller for recurrent connections
                    else:  # input-to-hidden
                        std = gamma
                else:
                    # Hidden layer weights - use gamma
                    std = gamma
                
                nn.init.normal_(param, mean=mean, std=std)
            else:
                # 1D parameters (bias, etc.) - skip or use default
                pass

def ordered_sweep(network, ranked_inputs):
    """
    Run network on ordered inputs for interpretable results.
    
    This function now works with any network implementing InterferenceTaskNetwork interface.
    For RNNs, it resets the hidden state before processing.
    
    Args:
        network: Network instance implementing InterferenceTaskNetwork
        ranked_inputs: Input tensor (n_samples, input_dim) or numpy array
        
    Returns:
        preds: Network predictions (n_samples, output_dim)
        hids: Hidden states (n_samples, hidden_dim)
    """
    # Convert to torch tensor if needed
    if isinstance(ranked_inputs, np.ndarray):
        ranked_inputs = torch.from_numpy(ranked_inputs).float()
    
    # Reset hidden state for RNNs
    if network.supports_sequences:
        batch_size = ranked_inputs.shape[0]
        network.reset_hidden_state(batch_size=batch_size)
    
    # Forward pass using interface method
    preds, hids = network(ranked_inputs)
    
    # Convert to numpy
    preds_np = preds.detach().cpu().numpy().copy()
    hids_np = hids.detach().cpu().numpy().copy()
    
    # Handle multi-dimensional hidden states (e.g., from RNN with multiple layers)
    if hids_np.ndim > 2:
        # For RNN: (num_layers, batch_size, hidden_dim) -> take last layer
        if hasattr(network, '_num_layers') and hids_np.shape[0] == network._num_layers:
            hids_np = hids_np[-1]  # Take last layer
        else:
            # Flatten or reshape appropriately
            hids_np = hids_np.reshape(hids_np.shape[0], -1)
    
    return preds_np, hids_np



"""
MAIN FUNCTION
"""
def main():
    # Set random seed
    set_seed(2024)
    condition_name = "rich_50" # Condition to run (e.g., rich_10, rich_50, rich_200)
    base_folder='./'
    
    # Network configuration - can be set via command line args or config file in future
    network_type = "ffn"  # Options: "ffn" or "rnn"
    rnn_cell_type = "RNN"  # Options: "RNN", "GRU", "LSTM" (only used if network_type == "rnn")
    rnn_num_layers = 1  # Number of RNN layers (only used if network_type == "rnn")
    rnn_dropout = 0.0  # Dropout probability (only used if network_type == "rnn")
    sequence_mode = "single"  # Options: "single" (each trial is seq_len=1) or "sequence" (only used if network_type == "rnn")

    # Setup paths
    data_folder = os.path.join(base_folder, 'data')
    config_path = os.path.join(base_folder, 'src', 'models', 'ann_experiments.json')

    # Load settings and find specified condition
    settings = json.load(open(config_path, 'r'))

    condition = next((c for c in settings['conditions'] if c['name'] == condition_name), None)
    if not condition:
        raise ValueError(f"Condition '{condition_name}' not found in settings")

    # Load participant data
    df = pd.read_csv(os.path.join(data_folder, 'participants', 'trial_df.csv'))

    df.loc[df['task_section']=='B','test_trial']=0

    df = df.loc[(df['task_section']=='A1') | 
                    (df['task_section']=='B') | 
                    (df['task_section']=='A2'), :] # remove debrief trials from analysis

    participants = df['participant'].unique()


    """
    SETUP
    """

    # Setup parameters
    task_parameters = {
            "nStim_perTask": 6,
            "schedules": ['same', 'near', 'far'],
            "schedule_names": ['same rule', 'near rule', 'far rule']
        }

    # Network parameters
    dim_input = task_parameters['nStim_perTask'] * 2
    dim_hidden = condition['dim_hidden']
    dim_output = 4  # 2 dimensions for each feature
    network_params = [dim_input, dim_hidden, dim_output]


    # Training parameters - convert to list format expected by neural_network.py
    training_params = [
        participants,  # participants list
        settings['n_phase'],  # n_phase
        settings['n_epochs'],  # n_epochs
        settings['n_epochs'] * (task_parameters['nStim_perTask']*2) * 10,  # n_train_trials
        settings['shuffle'],  # shuffle
        settings['batch_size'],  # batch_size
        condition['gamma'],  # gamma
        settings['learning_rate'],  # learning rate
    ]


    # Setup simulation folder if not existent yet
    sim_folder = os.path.join(data_folder, 'any_network', condition_name)
    os.makedirs(sim_folder, exist_ok=True)

    # Save settings of the run
    settings_to_save = {
        "condition": condition,
        "training_params": {
            "participants": ann.numpy_to_python(participants),
            "n_phase": settings['n_phase'],
            "n_epochs": settings['n_epochs'],
            "n_train_trials": settings['n_epochs'] * (task_parameters['nStim_perTask']*2) * 10,
            "shuffle": settings['shuffle'],
            "batch_size": settings['batch_size'],
            "gamma": condition['gamma'],
            "lr": settings['learning_rate'],
        },
        "network_params": network_params,
        "task_parameters": task_parameters,
        "network_config": {
            "network_type": network_type,
            "rnn_cell_type": rnn_cell_type if network_type.lower() == "rnn" else None,
            "rnn_num_layers": rnn_num_layers if network_type.lower() == "rnn" else None,
            "rnn_dropout": rnn_dropout if network_type.lower() == "rnn" else None,
            "sequence_mode": sequence_mode if network_type.lower() == "rnn" else None,
        }
    }

    # Convert numpy arrays to Python native types before saving
    settings_to_save = ann.numpy_to_python(settings_to_save)
    with open(os.path.join(sim_folder, 'settings.json'), 'w') as f:
        json.dump(settings_to_save, f, indent=4)


    # Unpack parameters
    dim_input, dim_hidden, dim_output = network_params
    participants, n_phase, n_epochs, n_train_trials, shuffle, batch_size, gamma, lr = training_params

    # add these params
    do_test = 1
    dosave=1

    results = []

    """
    TRAIN
    """

    # for each participant
    participant_results = {}

    for idx_p, participant in tqdm(enumerate(participants[0:10])): # test for first participant only
        print(f'Starting participant {participant}')

        # Get participant data
        dataset_A1, dataset_B, dataset_A2, raw_inputs, raw_labels = basic.get_datasets(df, participant, task_parameters)
        
        # Order inputs by feature
        A_inputs = raw_inputs[0]
        B_inputs = raw_inputs[1] 
        A_labels_feat1 = raw_labels[0, 0:2].T
        B_labels_feat1 = raw_labels[1, 0:2].T
        ordered_indices_A = basic.get_clockwise_order(A_labels_feat1)
        ordered_indices_B = basic.get_clockwise_order(B_labels_feat1)
        ordered_inputs = np.concatenate((A_inputs[ordered_indices_A], B_inputs[ordered_indices_B]), axis=0)

        # Create data loaders
        trainloader_A1 = DataLoader(CreateParticipantDataset(dataset_A1), batch_size=batch_size, shuffle=shuffle)
        trainloader_B = DataLoader(CreateParticipantDataset(dataset_B), batch_size=batch_size, shuffle=shuffle)
        trainloader_A2 = DataLoader(CreateParticipantDataset(dataset_A2), batch_size=batch_size, shuffle=shuffle)


        # Run a complete learning cycle
        """
        Runs a complete learning cycle:
        A: n_epochs of training on task A stimuli
        B: n_epochs of training on task B stimuli
        """
        n_train_trials = n_epochs * dim_input * 10
        n_phase = 3  # A, B, A

        # Preallocate results matrices
        results = {
            "indexes": np.full((n_phase, n_train_trials), np.nan, dtype=np.float32),
            "inputs": np.full((n_phase, n_train_trials, dim_input), np.nan, dtype=np.float32),
            "labels": np.full((n_phase, n_train_trials, 2), np.nan, dtype=np.float32),
            "test_stim": np.full((n_phase, n_train_trials), np.nan, dtype=np.float32),
            "probes": np.full((n_phase, n_train_trials), np.nan, dtype=np.float32),
            "losses": np.full((n_phase, n_train_trials), np.nan, dtype=np.float32),
            "accuracy": np.full((n_phase, n_train_trials), np.nan, dtype=np.float32),
            "predictions": np.full((n_phase, n_train_trials, dim_output), np.nan, dtype=np.float32),
            "hiddens": np.full((n_phase, n_train_trials, dim_hidden), np.nan, dtype=np.float32),
            "embeddings": np.full((n_phase, n_train_trials, dim_hidden, dim_input), np.nan, dtype=np.float32),
            "readouts": np.full((n_phase, n_train_trials, dim_output, dim_hidden), np.nan, dtype=np.float32),
        }

        # Define the network based on network_type
        if network_type.lower() == "ffn":
            network = simpleLinearNet(dim_input, dim_hidden, dim_output)
        elif network_type.lower() == "rnn":
            network = SimpleRNN(
                dim_input=dim_input,
                dim_hidden=dim_hidden,
                dim_output=dim_output,
                num_layers=rnn_num_layers,
                cell_type=rnn_cell_type,
                dropout=rnn_dropout,
                sequence_mode=sequence_mode
            )
        else:
            raise ValueError(f"Unsupported network_type: {network_type}. Must be 'ffn' or 'rnn'.")

        # Initialize weights
        ex_initializer_(network, gamma)

        optimizer = torch.optim.SGD(network.parameters(), lr=lr)
        loss_function = nn.MSELoss()

        # Initial pass of the network
        initial_preds, initial_hiddens = ordered_sweep(network, torch.from_numpy(ordered_inputs).float())

        results["preds_pre_training"] = initial_preds
        results["hiddens_pre_training"] = initial_hiddens

        
        # Training Phases
        phases = [
            (0, trainloader_A1, 1),
            (1, trainloader_B, 1),
            (2, trainloader_A2, 2),
        ]

        # Phase A1
        phase = 0
        loader =  trainloader_A1
        do_update = 1 # Controls how updates are applied (0 = no update, 1 = standard, 2 = conditional on feature_probe).

        (
            results["indexes"][phase, :],
            results["inputs"][phase, :, :],
            results["labels"][phase, :, :],
            results["probes"][phase, :],
            results["test_stim"][phase, :],
            results["losses"][phase, :],
            results["accuracy"][phase, :],
            results["predictions"][phase, :, :],
            results["hiddens"][phase, :, :],
            results["embeddings"][phase, :, :, :],
            results["readouts"][phase, :, :, :],
        ) = train_participant_schedule(
            network, loader, n_epochs, loss_function, optimizer, do_update, do_test
        )

        # Post-phase ordered sweep
        post_preds, post_hiddens = ordered_sweep(network, torch.from_numpy(ordered_inputs).float())
        results[f"preds_post_phase_{phase}"] = post_preds
        results[f"hiddens_post_phase_{phase}"] = post_hiddens

        phase = 1
        loader =  trainloader_B
        do_update = 1 # Controls how updates are applied (0 = no update, 1 = standard, 2 = conditional on feature_probe).

        (
            results["indexes"][phase, :],
            results["inputs"][phase, :, :],
            results["labels"][phase, :, :],
            results["probes"][phase, :],
            results["test_stim"][phase, :],
            results["losses"][phase, :],
            results["accuracy"][phase, :],
            results["predictions"][phase, :, :],
            results["hiddens"][phase, :, :],
            results["embeddings"][phase, :, :, :],
            results["readouts"][phase, :, :, :],
        ) = train_participant_schedule(
            network, loader, n_epochs, loss_function, optimizer, do_update, do_test
        )


        # Post-phase ordered sweep
        post_preds, post_hiddens = ordered_sweep(network, torch.from_numpy(ordered_inputs).float())
        results[f"preds_post_phase_{phase}"] = post_preds
        results[f"hiddens_post_phase_{phase}"] = post_hiddens


        phase = 2
        loader =  trainloader_A2
        do_update = 2  # Controls how updates are applied (0 = no update, 1 = standard, 2 = conditional on feature_probe).

        (
            results["indexes"][phase, :],
            results["inputs"][phase, :, :],
            results["labels"][phase, :, :],
            results["probes"][phase, :],
            results["test_stim"][phase, :],
            results["losses"][phase, :],
            results["accuracy"][phase, :],
            results["predictions"][phase, :, :],
            results["hiddens"][phase, :, :],
            results["embeddings"][phase, :, :, :],
            results["readouts"][phase, :, :, :],
        ) = train_participant_schedule(
            network, loader, n_epochs, loss_function, optimizer, do_update, do_test
        )

        # Post-phase ordered sweep
        post_preds, post_hiddens = ordered_sweep(network, torch.from_numpy(ordered_inputs).float())
        results[f"preds_post_phase_{phase}"] = post_preds
        results[f"hiddens_post_phase_{phase}"] = post_hiddens

        # Store participant with the accompanying results
        results['participant'] = participant

        participant_results[idx_p] = results

    """
    SAVE RESULTS 
    # commented out for now to avoid saving results
    # Save results if requested
    if dosave:
        file_path = f"{sim_folder}/sim_{participant}.npz"
        np.savez_compressed(file_path, **results)

    # Cleanup
    del participant_results
    """


if __name__ == "__main__":
    main()