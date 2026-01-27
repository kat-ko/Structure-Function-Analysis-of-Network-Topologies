"""
Neural network models

This module contains the core functionality for training neural networks on participant schedules.

Key components:
- simpleLinearNet: Neural network architecture
- CreateParticipantDataset: Dataset class for loading participant data
- Training utilities: Functions for training and evaluating the network
"""
import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from src.utils import basic_funcs as basic
import math
import copy
from tqdm.auto import tqdm

class simpleLinearNet(nn.Module):
    """A simple linear neural network with one hidden layer.
    
    Architecture:
    input -> hidden layer -> output
    All layers are fully connected with no bias terms.
    """
    def __init__(self, dim_input, dim_hidden, dim_output):
        super(simpleLinearNet, self).__init__()
        self.in_hid = nn.Linear(dim_input, dim_hidden, bias=False)
        self.hid_out = nn.Linear(dim_hidden, dim_output, bias=False)
        
    def forward(self, x, hidden=None, task_id=None):
        """Forward pass through the network.
        
        Args:
            x: Input tensor
            hidden: Optional hidden state (ignored for this network, for interface compatibility)
            task_id: Optional task identifier (ignored for this network, for interface compatibility)
        
        Returns:
            Tuple of (output, hidden_state)
        """
        hid = self.in_hid(x)
        out = self.hid_out(hid)
        return out, hid
    
    def get_embeddings(self):
        """Get input embedding weights (input-to-hidden layer)."""
        return self.in_hid.weight
    
    def get_readouts(self):
        """Get output readout weights (hidden-to-output layer)."""
        return self.hid_out.weight

def ex_initializer_(model, gamma=1e-3, mean=0.0, init_type="custom"):
    """
    In-place Re-initialization of weights

    Args:
        model: torch.nn.Module
        PyTorch neural net model
        
        gamma: float
        Initialization scale (used when init_type="custom")
        
        mean: float
        Mean for weight initialization (used when init_type="custom")
        
        init_type: str
        Type of initialization: "custom" (gamma-based) or "standard" (Xavier/Glorot)

    Returns:
        Nothing
    """
    for name, param in model.named_parameters():
        if "weight" in name:  
            n_out, n_in = param.shape
            
            if init_type == "standard":
                # Use Xavier/Glorot initialization for standard regime
                if "hid_out" in name:  # Output layer weights
                    # Keep small initialization for output layer
                    nn.init.normal_(param, mean=0.0, std=1e-3)
                else:  # Hidden layer weights
                    # Use Xavier/Glorot normal initialization
                    nn.init.xavier_normal_(param, gain=1.0)
            else:  # init_type == "custom" (default)
                # Use existing gamma-based initialization
                if "hid_out" in name:  # Output layer weights
                    std = 1e-3
                elif "comm_ab" in name.lower() or "comm_ba" in name.lower():  # Communication layers
                    std = 1e-3
                elif "readout_a" in name.lower() or "readout_b" in name.lower():  # Readout layers
                    std = 1e-3
                elif "mod_a" in name.lower() or "mod_b" in name.lower():  # Module input layers
                    std = gamma
                else:  # Hidden layer weights
                    std = gamma
                    
                nn.init.normal_(param, mean=mean, std=std)

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

def ordered_sweep(network, ranked_inputs, n_stim_per_task=6):
    """
    Run network on ordered inputs for interpretable results.
    
    For two-module networks, this handles task routing based on input position.
    """
    from src.models.two_module_mlp import TwoModuleMLP
    
    # Convert to torch tensor if needed
    if isinstance(ranked_inputs, np.ndarray):
        ranked_inputs = torch.from_numpy(ranked_inputs).float()
    
    is_two_module = isinstance(network, TwoModuleMLP)
    
    if is_two_module:
        # For two-module networks, process with task routing
        n_samples = ranked_inputs.shape[0]
        preds_list = []
        hids_list = []
        h_A_list = []
        h_B_list = []
        
        for i in range(n_samples):
            input_sample = ranked_inputs[i:i+1]
            # Determine task_id based on position
            if i < n_stim_per_task:
                task_id = "A"
            else:
                task_id = "B"
            
            # Forward pass with task_id
            pred, hid = network(input_sample, task_id=task_id)
            preds_list.append(pred)
            hids_list.append(hid)
            
            # Store per-module hidden states
            try:
                h_A, h_B = network.get_module_hidden_states()
                h_A_list.append(h_A)
                h_B_list.append(h_B)
            except RuntimeError:
                h_A_list.append(torch.zeros(1, network.dim_hidden_per_module, device=pred.device))
                h_B_list.append(torch.zeros(1, network.dim_hidden_per_module, device=pred.device))
        
        preds = torch.cat(preds_list, dim=0)
        hids = torch.cat(hids_list, dim=0)
        network._ordered_sweep_h_A = torch.cat(h_A_list, dim=0)
        network._ordered_sweep_h_B = torch.cat(h_B_list, dim=0)
    else:
        # For single-module networks, process all at once
        preds, hids = network(ranked_inputs)
    
    return preds.detach().numpy().copy(), hids.detach().numpy().copy()

def run_simulation(training_params, network_params, task_parameters, df, do_test, dosave=0, sim_folder=np.nan, init_type="custom", architecture_config=None):
    """Run neural network simulation for participant learning.
    
    1. Initializes network and loads participant data
    2. Trains network on sequence: A1 -> B -> A2
    3. Records and optionally saves results
    
    Args:
        training_params: Parameters for training (participants, epochs, etc)
        network_params: Network architecture parameters
        task_parameters: Task-specific parameters
        df: DataFrame with participant data
        do_test: Whether to run test trials
        dosave: Whether to save results
        sim_folder: Folder to save results if dosave=1
        init_type: Type of initialization ("custom" or "standard")
        architecture_config: Optional architecture configuration dict
            If type == 'two_module', creates TwoModuleMLP instead of simpleLinearNet
        
    Returns:
        List of results per participant
    """
    # Unpack parameters
    dim_input, dim_hidden, dim_output = network_params
    participants, n_phase, n_epochs, n_train_trials, shuffle, batch_size, gamma, lr = training_params
    
    results = []
    
    # Train network for each participant
    for idx_p, participant in tqdm(enumerate(participants)):
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

        # Train network through phases A1 -> B -> A2
        participant_results = runSchedule(
            train_participant_schedule, lr, gamma, n_epochs, dim_input, dim_hidden,
            dim_output, trainloader_A1, trainloader_B, trainloader_A2, ordered_inputs, do_test, init_type, architecture_config
        )

        participant_results['participant'] = participant
        
        # Save results if requested
        if dosave:
            file_path = f"{sim_folder}/sim_{participant}.npz"
            np.savez_compressed(file_path, **participant_results)
        
        # Cleanup
        del participant_results

    return results

def runSchedule(train_function, lr, gamma, n_epochs, dim_input, dim_hidden, dim_output, trainloader_A1, trainloader_B, trainloader_A2, ordered_inputs, do_test, init_type="custom", architecture_config=None):
    """
    Runs a complete learning cycle:
    A: n_epochs of training on task A stimuli
    B: n_epochs of training on task B stimuli
    
    Args:
        init_type: Type of initialization ("custom" or "standard")
        architecture_config: Optional architecture configuration dict with 'type' key
            If type == 'two_module', creates TwoModuleMLP instead of simpleLinearNet
    """
    n_train_trials = n_epochs * dim_input * 10
    n_phase = 3  # A, B, A

    # Check if this is a two-module architecture
    is_two_module = (architecture_config is not None and 
                     architecture_config.get('type') == 'two_module')
    
    if is_two_module:
        dim_hidden_per_module = dim_hidden // 2
    else:
        dim_hidden_per_module = dim_hidden

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
    
    # Add per-module hidden state storage if two-module
    if is_two_module:
        results["hiddens_A"] = np.full((n_phase, n_train_trials, dim_hidden_per_module), np.nan, dtype=np.float32)
        results["hiddens_B"] = np.full((n_phase, n_train_trials, dim_hidden_per_module), np.nan, dtype=np.float32)

    # Define the network based on architecture type
    if is_two_module:
        from src.models.two_module_mlp import TwoModuleMLP
        comm_bandwidth = architecture_config.get('comm_bandwidth', 'none')
        comm_scale = architecture_config.get('comm_scale', None)
        network = TwoModuleMLP(dim_input, dim_hidden, dim_output, 
                              comm_bandwidth=comm_bandwidth, comm_scale=comm_scale)
    else:
        network = simpleLinearNet(dim_input, dim_hidden, dim_output)

    # Initialize weights
    ex_initializer_(network, gamma, init_type=init_type)

    optimizer = torch.optim.SGD(network.parameters(), lr=lr)
    loss_function = nn.MSELoss()
  
    # Initial pass of the network
    n_stim_per_task = ordered_inputs.shape[0] // 2
    initial_preds, initial_hiddens = ordered_sweep(network, torch.from_numpy(ordered_inputs).float(), n_stim_per_task=n_stim_per_task)
    results["preds_pre_training"] = initial_preds
    results["hiddens_pre_training"] = initial_hiddens
    
    # Store per-module hidden states for pre-training if available
    if is_two_module and hasattr(network, '_ordered_sweep_h_A') and hasattr(network, '_ordered_sweep_h_B'):
        results["hiddens_A_pre_training"] = network._ordered_sweep_h_A.detach().cpu().numpy()
        results["hiddens_B_pre_training"] = network._ordered_sweep_h_B.detach().cpu().numpy()

    # Training Phases
    phases = [
        (0, trainloader_A1, 1),
        (1, trainloader_B, 1),
        (2, trainloader_A2, 2),
    ]
    for phase, loader, do_update in phases:
        train_results = train_function(
            network, loader, n_epochs, loss_function, optimizer, do_update, do_test
        )
        
        # Unpack results (handle both old and new return formats)
        if len(train_results) >= 13:
            # New format with hiddens_A and hiddens_B
            (results["indexes"][phase, :],
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
             hiddens_A,
             hiddens_B) = train_results
            # Store per-module hidden states if available and results dict has the keys
            if is_two_module and "hiddens_A" in results and len(hiddens_A) > 0:
                n_samples = min(len(hiddens_A), results["hiddens_A"].shape[1])
                if n_samples > 0:
                    results["hiddens_A"][phase, :n_samples, :] = hiddens_A[:n_samples]
            if is_two_module and "hiddens_B" in results and len(hiddens_B) > 0:
                n_samples = min(len(hiddens_B), results["hiddens_B"].shape[1])
                if n_samples > 0:
                    results["hiddens_B"][phase, :n_samples, :] = hiddens_B[:n_samples]
        else:
            # Old format (backward compatibility)
            (results["indexes"][phase, :],
             results["inputs"][phase, :, :],
             results["labels"][phase, :, :],
             results["probes"][phase, :],
             results["test_stim"][phase, :],
             results["losses"][phase, :],
             results["accuracy"][phase, :],
             results["predictions"][phase, :, :],
             results["hiddens"][phase, :, :],
             results["embeddings"][phase, :, :, :],
             results["readouts"][phase, :, :, :]) = train_results

        # Post-phase ordered sweep
        n_stim_per_task = ordered_inputs.shape[0] // 2
        post_preds, post_hiddens = ordered_sweep(network, torch.from_numpy(ordered_inputs).float(), n_stim_per_task=n_stim_per_task)
        results[f"preds_post_phase_{phase}"] = post_preds
        results[f"hiddens_post_phase_{phase}"] = post_hiddens
        
        # Store per-module hidden states if available
        if is_two_module and hasattr(network, '_ordered_sweep_h_A') and hasattr(network, '_ordered_sweep_h_B'):
            h_A_np = network._ordered_sweep_h_A.detach().cpu().numpy()
            h_B_np = network._ordered_sweep_h_B.detach().cpu().numpy()
            if f"hiddens_A_post_phase_{phase}" not in results:
                # Initialize if not exists
                n_stim_total = ordered_inputs.shape[0]
                for p in range(n_phase):
                    results[f"hiddens_A_post_phase_{p}"] = np.full((n_stim_total, dim_hidden_per_module), np.nan, dtype=np.float32)
                    results[f"hiddens_B_post_phase_{p}"] = np.full((n_stim_total, dim_hidden_per_module), np.nan, dtype=np.float32)
            results[f"hiddens_A_post_phase_{phase}"] = h_A_np
            results[f"hiddens_B_post_phase_{phase}"] = h_B_np

    return results





def train_participant_schedule(network, trainloader, n_epochs, loss_function, optimizer, do_update, do_test):
    """
    Train the network on x-y coordinates 

    Args:
        network: The neural network to be trained.
        trainloader: DataLoader object containing training data.
        n_epochs: Number of epochs to train.
        loss_function: The loss function to use.
        optimizer: Optimizer for updating network parameters.
        do_update: Controls how updates are applied (0 = no update, 1 = standard, 2 = conditional on feature_probe).

    Returns:
        tuple: Various metrics including indexes, inputs, labels, probes, losses, accuracy, predictions, hiddens, embeddings, readouts.
    """
    # Initialize storage lists
    metrics = {
        "indexes": [],
        "losses": [],
        "accuracy": [],
        "predictions": [],
        "hiddens": [],
        "hiddens_A": [],  # Per-module hidden states for two-module networks
        "hiddens_B": [],
        "embeddings": [],
        "readouts": [],
        "probes": [],
        "test_stim":[],
        "labels": [],
        "inputs": [],
    }

    for epoch in range(n_epochs):
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
            task_id = data.get('task_id', None)  # Get task_id if available
            
            # Handle task_id: if it's a numpy array, get first element; if string, use as-is
            if task_id is not None:
                if isinstance(task_id, np.ndarray):
                    task_id = task_id[0] if task_id.size > 0 else None
                elif isinstance(task_id, (list, tuple)):
                    task_id = task_id[0] if len(task_id) > 0 else None
            
                    
            joined_label = torch.cat((label_x.unsqueeze(1), label_y.unsqueeze(1)), dim=1)
            radians_label = math.atan2(label_x, label_y)

            # Forward pass with task_id
            out, hid = network(input, task_id=task_id)
            
            # Extract per-module hidden states for two-module networks
            h_A = None
            h_B = None
            try:
                from src.models.two_module_mlp import TwoModuleMLP
                if isinstance(network, TwoModuleMLP):
                    h_A, h_B = network.get_module_hidden_states()
                    h_A = h_A.detach().cpu().numpy()
                    h_B = h_B.detach().cpu().numpy()
            except (RuntimeError, AttributeError):
                # If get_module_hidden_states fails or network doesn't have it, set to None
                pass

            # Calculate loss based on feature probe
            if feature_probe == 0:
                loss = loss_function(out[:, :2], joined_label)
                pred_rads = math.atan2(out[:, 0].detach().numpy(),out[:, 1].detach().numpy())
                accuracy = compute_accuracy(pred_rads, radians_label)
                
            elif feature_probe == 1:
                loss = loss_function(out[:, 2:4], joined_label)
                pred_rads = math.atan2(out[:, 2].detach().numpy(),out[:, 3].detach().numpy())
                accuracy = compute_accuracy(pred_rads, radians_label)
                
            else:
                raise ValueError("Undefined loss setting for feature_probe.")

            # Update network if required
            if do_update == 1 and do_test==1 and test_stim.numpy() == 0:
              loss.backward()
              optimizer.step()
            elif do_update == 1 and do_test ==0:
              loss.backward()
              optimizer.step()
            elif do_update == 2 and feature_probe == 0:  # In C, only update for feature 0 
              loss.backward()
              optimizer.step()

            # Store metrics
            metrics["indexes"].append(index)
            metrics["inputs"].append(input.numpy())
            metrics["labels"].append(joined_label.numpy())
            metrics["probes"].append(feature_probe.numpy())
            metrics["test_stim"].append(test_stim.numpy())
            metrics["losses"].append(loss.item())
            metrics["accuracy"].append(accuracy)
            metrics["predictions"].append(np.expand_dims(out.detach().numpy(), axis=1))
            metrics["hiddens"].append(hid.detach().numpy())
            
            # Store per-module hidden states if available
            if h_A is not None:
                metrics["hiddens_A"].append(h_A)
            else:
                metrics["hiddens_A"].append(None)
            if h_B is not None:
                metrics["hiddens_B"].append(h_B)
            else:
                metrics["hiddens_B"].append(None)
            
            # Get embeddings and readouts (handle both single and two-module networks)
            embeddings = network.get_embeddings()
            if embeddings is not None:
                metrics["embeddings"].append(embeddings.detach().cpu().numpy())
            else:
                metrics["embeddings"].append(None)
            
            readouts = network.get_readouts()
            if readouts is not None:
                metrics["readouts"].append(readouts.detach().cpu().numpy())
            else:
                metrics["readouts"].append(None)

    # Handle per-module hidden states (hiddens_A, hiddens_B)
    for key in ["hiddens_A", "hiddens_B"]:
        if metrics[key] and all(x is None for x in metrics[key]):
            metrics[key] = np.array([])
        elif metrics[key] and any(x is None for x in metrics[key]):
            # Filter out None values
            metrics[key] = [x for x in metrics[key] if x is not None]
            if metrics[key]:
                metrics[key] = np.squeeze(np.array(metrics[key]))
            else:
                metrics[key] = np.array([])
        elif metrics[key]:
            metrics[key] = np.squeeze(np.array(metrics[key]))
        else:
            metrics[key] = np.array([])
    
    # Handle embeddings and readouts (may contain None)
    for key in ["embeddings", "readouts"]:
        if metrics[key] and all(x is None for x in metrics[key]):
            metrics[key] = np.array([])
        elif metrics[key] and any(x is None for x in metrics[key]):
            # Filter out None values
            metrics[key] = [x for x in metrics[key] if x is not None]
            if metrics[key]:
                metrics[key] = np.squeeze(np.array(metrics[key]))
            else:
                metrics[key] = np.array([])
        elif metrics[key]:
            metrics[key] = np.squeeze(np.array(metrics[key]))
        else:
            metrics[key] = np.array([])
    
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
        metrics.get("hiddens_A", np.array([])),  # Per-module hidden states (empty for non-two-module)
        metrics.get("hiddens_B", np.array([])),  # Per-module hidden states (empty for non-two-module)
    )


def train_single_schedule(training_params, network_params, task_parameters, df, do_test, init_type="custom", architecture_config=None):
    """
    Train a single schedule for geometry visualization.
    
    Args:
        training_params: Training parameters
        network_params: Network architecture parameters
        task_parameters: Task-specific parameters
        df: DataFrame with participant data
        do_test: Whether to run test trials
        init_type: Type of initialization ("custom" or "standard")
        architecture_config: Optional architecture configuration dict
            If type == 'two_module', creates TwoModuleMLP instead of simpleLinearNet
    """
    dim_input, dim_hidden, dim_output = network_params
    _, n_phase, n_epochs, n_train_trials, shuffle, batch_size, gamma, lr = training_params
    
    # Check if this is a two-module architecture
    is_two_module = (architecture_config is not None and 
                     architecture_config.get('type') == 'two_module')
    
    if is_two_module:
        dim_hidden_per_module = dim_hidden // 2
    else:
        dim_hidden_per_module = dim_hidden
    
    # Phantom df created where all groups trained on same A, for geometry visualisation
    dataset_A1, dataset_B_same, dataset_A2, raw_inputs, raw_labels = basic.get_datasets(df, 'geom_sub_same', task_parameters)
    _, dataset_B_near, _, _, _ = basic.get_datasets(df, 'geom_sub_near', task_parameters)
    _, dataset_B_far, _, _, _ = basic.get_datasets(df, 'geom_sub_far', task_parameters)
    
    A_inputs = raw_inputs[0]
    B_inputs = raw_inputs[1]
    A_labels_feat1 = raw_labels[0, 0:2].T
    B_labels_feat1 = raw_labels[1, 0:2].T

    ordered_indices_A = basic.get_clockwise_order(A_labels_feat1)
    ordered_indices_B = basic.get_clockwise_order(B_labels_feat1)
    ordered_inputs = np.concatenate((A_inputs[ordered_indices_A], B_inputs[ordered_indices_B]), axis=0)

    trainloader_A1 = DataLoader(CreateParticipantDataset(dataset_A1), batch_size=batch_size, shuffle=shuffle)
    trainloader_B_same = DataLoader(CreateParticipantDataset(dataset_B_same), batch_size=batch_size, shuffle=shuffle)
    trainloader_B_near = DataLoader(CreateParticipantDataset(dataset_B_near), batch_size=batch_size, shuffle=shuffle)
    trainloader_B_far = DataLoader(CreateParticipantDataset(dataset_B_far), batch_size=batch_size, shuffle=shuffle)
    trainloader_A2 = DataLoader(CreateParticipantDataset(dataset_A2), batch_size=batch_size, shuffle=shuffle)

    
    n_train_trials = n_epochs * dim_input * 10
    n_phase = 3  # A, B, A

    # Preallocate results matrices

    results = {
        "indexes": np.full((3, n_phase, n_train_trials), np.nan, dtype=np.float32),
        "inputs": np.full((3, n_phase, n_train_trials, dim_input), np.nan, dtype=np.float32),
        "labels": np.full((3, n_phase, n_train_trials, 2), np.nan, dtype=np.float32),
        "test_stim": np.full((3, n_phase, n_train_trials), np.nan, dtype=np.float32),
        "probes": np.full((3, n_phase, n_train_trials), np.nan, dtype=np.float32),
        "losses": np.full((3, n_phase, n_train_trials), np.nan, dtype=np.float32),
        "accuracy": np.full((3, n_phase, n_train_trials), np.nan, dtype=np.float32),
        "predictions": np.full((3, n_phase, n_train_trials, dim_output), np.nan, dtype=np.float32),
        "hiddens": np.full((3, n_phase, n_train_trials, dim_hidden), np.nan, dtype=np.float32),
        "embeddings": np.full((3, n_phase, n_train_trials, dim_hidden, dim_input), np.nan, dtype=np.float32),
        "readouts": np.full((3, n_phase, n_train_trials, dim_output, dim_hidden), np.nan, dtype=np.float32),
        "preds_pre_training": np.full((3, task_parameters['nStim_perTask']*2, dim_output), np.nan, dtype=np.float32),
        "hiddens_pre_training": np.full((3, task_parameters['nStim_perTask']*2, dim_hidden), np.nan, dtype=np.float32),
        "preds_post_phase_0": np.full((3,  task_parameters['nStim_perTask']*2, dim_output), np.nan, dtype=np.float32),
        "hiddens_post_phase_0": np.full((3,  task_parameters['nStim_perTask']*2, dim_hidden), np.nan, dtype=np.float32),
        "preds_post_phase_1": np.full((3, task_parameters['nStim_perTask']*2, dim_output), np.nan, dtype=np.float32),
        "hiddens_post_phase_1": np.full((3,  task_parameters['nStim_perTask']*2, dim_hidden), np.nan, dtype=np.float32),
        "preds_post_phase_2": np.full((3,  task_parameters['nStim_perTask']*2, dim_output), np.nan, dtype=np.float32),
        "hiddens_post_phase_2": np.full((3,  task_parameters['nStim_perTask']*2, dim_hidden), np.nan, dtype=np.float32),

    }
    
    # Add per-module hidden state storage if two-module
    if is_two_module:
        n_stim_total = task_parameters['nStim_perTask'] * 2
        n_train_trials = n_epochs * dim_input * 10
        # Add per-module hidden states for training phases
        results["hiddens_A"] = np.full((3, n_phase, n_train_trials, dim_hidden_per_module), np.nan, dtype=np.float32)
        results["hiddens_B"] = np.full((3, n_phase, n_train_trials, dim_hidden_per_module), np.nan, dtype=np.float32)
        # Add per-module hidden states for ordered sweeps
        for phase in range(n_phase):
            results[f"hiddens_A_post_phase_{phase}"] = np.full((3, n_stim_total, dim_hidden_per_module), np.nan, dtype=np.float32)
            results[f"hiddens_B_post_phase_{phase}"] = np.full((3, n_stim_total, dim_hidden_per_module), np.nan, dtype=np.float32)
        results["hiddens_A_pre_training"] = np.full((3, n_stim_total, dim_hidden_per_module), np.nan, dtype=np.float32)
        results["hiddens_B_pre_training"] = np.full((3, n_stim_total, dim_hidden_per_module), np.nan, dtype=np.float32)

    # Define the network based on architecture type
    if is_two_module:
        from src.models.two_module_mlp import TwoModuleMLP
        comm_bandwidth = architecture_config.get('comm_bandwidth', 'none')
        comm_scale = architecture_config.get('comm_scale', None)
        network = TwoModuleMLP(dim_input, dim_hidden, dim_output, 
                              comm_bandwidth=comm_bandwidth, comm_scale=comm_scale)
    else:
        network = simpleLinearNet(dim_input, dim_hidden, dim_output)

    # Initialize weights
    ex_initializer_(network, gamma, init_type=init_type)

    optimizer = torch.optim.SGD(network.parameters(), lr=lr)
    loss_function = nn.MSELoss()
  
    # Initial pass of the network
    n_stim_per_task = ordered_inputs.shape[0] // 2
    initial_preds, initial_hiddens = ordered_sweep(network, torch.from_numpy(ordered_inputs).float(), n_stim_per_task=n_stim_per_task)
    results["preds_pre_training"] = initial_preds
    results["hiddens_pre_training"] = initial_hiddens
    
    # Store per-module hidden states for pre-training if available
    if is_two_module and hasattr(network, '_ordered_sweep_h_A') and hasattr(network, '_ordered_sweep_h_B'):
        results["hiddens_A_pre_training"] = network._ordered_sweep_h_A.detach().cpu().numpy()
        results["hiddens_B_pre_training"] = network._ordered_sweep_h_B.detach().cpu().numpy()
    
    train_results = train_participant_schedule(network, trainloader_A1, n_epochs, loss_function, optimizer, 1, do_test)
    # Unpack results (handle both old and new return formats)
    if len(train_results) >= 13:
        # New format with hiddens_A and hiddens_B
        (results["indexes"][0, 0, :],
         results["inputs"][0,0, :, :],
         results["labels"][0,0, :, :],
         results["probes"][0,0, :],
         results["test_stim"][0,0, :],
         results["losses"][0,0, :],
         results["accuracy"][0,0, :],
         results["predictions"][0,0, :, :],
         results["hiddens"][0,0, :, :],
         results["embeddings"][0,0, :, :, :],
         results["readouts"][0,0, :, :, :],
         hiddens_A,
         hiddens_B) = train_results
        # Store per-module hidden states if available
        if is_two_module and "hiddens_A" in results and len(hiddens_A) > 0:
            n_samples = min(len(hiddens_A), results["hiddens_A"].shape[2] if "hiddens_A" in results else 0)
            if n_samples > 0 and "hiddens_A" in results:
                results["hiddens_A"][0, 0, :n_samples, :] = hiddens_A[:n_samples]
        if is_two_module and "hiddens_B" in results and len(hiddens_B) > 0:
            n_samples = min(len(hiddens_B), results["hiddens_B"].shape[2] if "hiddens_B" in results else 0)
            if n_samples > 0 and "hiddens_B" in results:
                results["hiddens_B"][0, 0, :n_samples, :] = hiddens_B[:n_samples]
    else:
        # Old format (backward compatibility)
        (results["indexes"][0, 0, :],
         results["inputs"][0,0, :, :],
         results["labels"][0,0, :, :],
         results["probes"][0,0, :],
         results["test_stim"][0,0, :],
         results["losses"][0,0, :],
         results["accuracy"][0,0, :],
         results["predictions"][0,0, :, :],
         results["hiddens"][0,0, :, :],
         results["embeddings"][0,0, :, :, :],
         results["readouts"][0,0, :, :, :]) = train_results
    
    # Post-phase ordered sweep
    n_stim_per_task = ordered_inputs.shape[0] // 2
    post_preds, post_hiddens = ordered_sweep(network, torch.from_numpy(ordered_inputs).float(), n_stim_per_task=n_stim_per_task)
    results[f"preds_post_phase_0"][0,:,:] = post_preds
    results[f"hiddens_post_phase_0"][0,:,:] = post_hiddens
    
    # Store per-module hidden states if available
    if is_two_module and hasattr(network, '_ordered_sweep_h_A') and hasattr(network, '_ordered_sweep_h_B'):
        h_A_np = network._ordered_sweep_h_A.detach().cpu().numpy()
        h_B_np = network._ordered_sweep_h_B.detach().cpu().numpy()
        if "hiddens_A_post_phase_0" in results:
            results["hiddens_A_post_phase_0"][0, :, :] = h_A_np
        if "hiddens_B_post_phase_0" in results:
            results["hiddens_B_post_phase_0"][0, :, :] = h_B_np
     
    # Now split the network into the three sessions 
    network_same = copy.deepcopy(network)
    network_near = copy.deepcopy(network)
    network_far = copy.deepcopy(network)
    optimizer_same = torch.optim.SGD(network_same.parameters(), lr=lr)
    optimizer_near = torch.optim.SGD(network_near.parameters(), lr=lr)
    optimizer_far = torch.optim.SGD(network_far.parameters(), lr=lr)
        
    for condition_idx, (condition_network, condition_training, condition_optimizer) in enumerate(zip(
    [network_same, network_near, network_far],
    [trainloader_B_same, trainloader_B_near, trainloader_B_far],
    [optimizer_same, optimizer_near, optimizer_far])): 
         
        phases = [
            (1, condition_training, 1),
            (2, trainloader_A2, 2),
        ]
         
        for phase, loader, do_update in phases:
            train_results = train_participant_schedule(condition_network, loader, n_epochs, loss_function, condition_optimizer, do_update, do_test)
            
            # Unpack results (handle both old and new return formats)
            if len(train_results) >= 13:
                # New format with hiddens_A and hiddens_B
                (results["indexes"][condition_idx, phase, :],
                 results["inputs"][condition_idx, phase, :, :],
                 results["labels"][condition_idx, phase, :, :],
                 results["probes"][condition_idx, phase, :],
                 results["test_stim"][condition_idx, phase, :],
                 results["losses"][condition_idx, phase, :],
                 results["accuracy"][condition_idx, phase, :],
                 results["predictions"][condition_idx, phase, :, :],
                 results["hiddens"][condition_idx, phase, :, :],
                 results["embeddings"][condition_idx, phase, :, :, :],
                 results["readouts"][condition_idx, phase, :, :, :],
                 hiddens_A,
                 hiddens_B) = train_results
                # Store per-module hidden states if available
                if is_two_module and "hiddens_A" in results and len(hiddens_A) > 0:
                    n_samples = min(len(hiddens_A), results["hiddens_A"].shape[2] if "hiddens_A" in results else 0)
                    if n_samples > 0 and "hiddens_A" in results:
                        results["hiddens_A"][condition_idx, phase, :n_samples, :] = hiddens_A[:n_samples]
                if is_two_module and "hiddens_B" in results and len(hiddens_B) > 0:
                    n_samples = min(len(hiddens_B), results["hiddens_B"].shape[2] if "hiddens_B" in results else 0)
                    if n_samples > 0 and "hiddens_B" in results:
                        results["hiddens_B"][condition_idx, phase, :n_samples, :] = hiddens_B[:n_samples]
            else:
                # Old format (backward compatibility)
                (results["indexes"][condition_idx, phase, :],
                 results["inputs"][condition_idx, phase, :, :],
                 results["labels"][condition_idx, phase, :, :],
                 results["probes"][condition_idx, phase, :],
                 results["test_stim"][condition_idx, phase, :],
                 results["losses"][condition_idx, phase, :],
                 results["accuracy"][condition_idx, phase, :],
                 results["predictions"][condition_idx, phase, :, :],
                 results["hiddens"][condition_idx, phase, :, :],
                 results["embeddings"][condition_idx, phase, :, :, :],
                 results["readouts"][condition_idx, phase, :, :, :]) = train_results

            # Post-phase ordered sweep
            n_stim_per_task = ordered_inputs.shape[0] // 2
            post_preds, post_hiddens = ordered_sweep(condition_network, torch.from_numpy(ordered_inputs).float(), n_stim_per_task=n_stim_per_task)
            results[f"preds_post_phase_{phase}"][condition_idx,:,:] = post_preds
            results[f"hiddens_post_phase_{phase}"][condition_idx,:,:] = post_hiddens
            
            # Store per-module hidden states if available
            if is_two_module and hasattr(condition_network, '_ordered_sweep_h_A') and hasattr(condition_network, '_ordered_sweep_h_B'):
                h_A_np = condition_network._ordered_sweep_h_A.detach().cpu().numpy()
                h_B_np = condition_network._ordered_sweep_h_B.detach().cpu().numpy()
                if f"hiddens_A_post_phase_{phase}" in results:
                    results[f"hiddens_A_post_phase_{phase}"][condition_idx, :, :] = h_A_np
                if f"hiddens_B_post_phase_{phase}" in results:
                    results[f"hiddens_B_post_phase_{phase}"][condition_idx, :, :] = h_B_np
            
            
   
    return results