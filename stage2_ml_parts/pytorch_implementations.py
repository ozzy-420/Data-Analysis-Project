import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import time

# Configure logging
logger = logging.getLogger(__name__)


# Removed basicConfig from here as it's handled in logging_config.py and imported in __init__.py

class PyTorchLogisticRegression(nn.Module):
    """
    PyTorch implementation of Logistic Regression.
    """

    def __init__(self, input_dim):
        super(PyTorchLogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        """
        Forward pass for the model.
        Args:
            x (torch.Tensor): Input features.
        Returns:
            torch.Tensor: Logits.
        """
        return self.linear(x)


def train_pytorch_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=100, verbose=False):
    """
    Trains a PyTorch model and evaluates it on the validation set.

    Args:
        model (nn.Module): PyTorch model.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        device (torch.device): Device to train on (CPU/GPU).
        epochs (int): Number of epochs.
        verbose (bool): Whether to print progress.

    Returns:
        dict: Training history containing loss and accuracy.
    """
    model.to(device)
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': [], 'val_roc_auc': []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset) if len(train_loader.dataset) > 0 else 0
        history['train_loss'].append(epoch_loss)

        # Validation
        model.eval()
        val_running_loss = 0.0
        correct = 0
        total = 0
        all_val_labels_list = []
        all_val_outputs_list = []  # For ROC AUC
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * inputs.size(0)

                predicted_proba = torch.sigmoid(outputs)
                predicted = (predicted_proba > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_val_labels_list.extend(labels.cpu().numpy().flatten().tolist())
                all_val_outputs_list.extend(predicted_proba.cpu().numpy().flatten().tolist())

        val_epoch_loss = val_running_loss / len(val_loader.dataset) if len(val_loader.dataset) > 0 else 0
        val_epoch_acc = correct / total if total > 0 else 0
        history['val_loss'].append(val_epoch_loss)
        history['val_accuracy'].append(val_epoch_acc)

        # Calculate ROC AUC for validation set per epoch
        val_roc_auc = "N/A"
        if len(all_val_labels_list) > 0 and len(np.unique(all_val_labels_list)) > 1:
            try:
                val_roc_auc = roc_auc_score(all_val_labels_list, all_val_outputs_list)
            except ValueError as e:
                logger.debug(f"Could not compute ROC AUC for validation epoch {epoch + 1}: {e}")
        history['val_roc_auc'].append(val_roc_auc)

        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            logger.info(
                f"Epoch {epoch + 1}/{epochs} - Train Loss: {epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}, Val ROC_AUC: {val_roc_auc if isinstance(val_roc_auc, str) else val_roc_auc:.4f}")
    return history


def evaluate_pytorch_model(model, data_loader, device, criterion=None):
    """
    Evaluates a PyTorch model on a given dataset.

    Args:
        model (nn.Module): PyTorch model.
        data_loader (DataLoader): DataLoader for evaluation data.
        device (torch.device): Device to evaluate on (CPU/GPU).
        criterion (nn.Module, optional): Loss function.

    Returns:
        dict: Evaluation metrics including accuracy, F1, precision, recall, and ROC AUC.
    """
    model.to(device)
    model.eval()
    all_labels = []
    all_predictions = []
    all_probas = []
    total_loss = 0

    if len(data_loader.dataset) == 0:  # Handle empty dataset
        logger.warning("Evaluation on empty dataset. Returning N/A metrics.")
        return {
            "Accuracy": "N/A", "F1": "N/A", "Precision": "N/A",
            "Recall": "N/A", "ROC_AUC": "N/A", "Loss": "N/A"
        }

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(inputs)

            if criterion:
                loss = criterion(outputs, labels)
                total_loss += loss.item() * inputs.size(0)

            probas = torch.sigmoid(outputs)
            predicted = (probas > 0.5).float()

            all_labels.extend(labels.cpu().numpy().flatten())
            all_predictions.extend(predicted.cpu().numpy().flatten())
            all_probas.extend(probas.cpu().numpy().flatten())

    avg_loss = total_loss / len(data_loader.dataset) if criterion and len(data_loader.dataset) > 0 else None

    # Ensure all_labels is not empty and has more than one class for ROC AUC
    roc_auc_val = "N/A"
    if all_labels and len(np.unique(all_labels)) > 1:
        try:
            roc_auc_val = roc_auc_score(all_labels, all_probas)
        except ValueError as e:  # Catch errors like "Only one class present in y_true"
            logger.debug(f"Cannot compute ROC AUC during evaluation: {e}")
            roc_auc_val = "N/A"
    elif not all_labels:
        logger.warning("No labels found for evaluation, ROC AUC cannot be computed.")
        roc_auc_val = "N/A"
    else:  # Single class present
        logger.warning(f"Only one class ({np.unique(all_labels)}) present in labels, ROC AUC is 'N/A'.")
        roc_auc_val = "N/A"

    metrics = {
        "Accuracy": accuracy_score(all_labels, all_predictions) if all_labels else "N/A",
        "F1": f1_score(all_labels, all_predictions, zero_division=0) if all_labels else "N/A",
        "Precision": precision_score(all_labels, all_predictions, zero_division=0) if all_labels else "N/A",
        "Recall": recall_score(all_labels, all_predictions, zero_division=0) if all_labels else "N/A",
        "ROC_AUC": roc_auc_val
    }
    if avg_loss is not None:
        metrics["Loss"] = avg_loss
    return metrics


def run_pytorch_logistic_regression(X_train_np, y_train_np, X_val_np, y_val_np, X_test_np, y_test_np, input_dim,
                                    epochs=50, lr=0.01, batch_size=64):
    """
    Runs PyTorch logistic regression training and evaluation on CPU and GPU (if available).

    Args:
        X_train_np, X_val_np, X_test_np (np.ndarray): Feature matrices for training, validation, and testing.
        y_train_np, y_val_np, y_test_np (np.ndarray): Target vectors for training, validation, and testing.
        input_dim (int): Number of input features.
        epochs (int): Number of epochs.
        lr (float): Learning rate.
        batch_size (int): Batch size.

    Returns:
        tuple: (metrics_cpu, metrics_gpu, time_cpu, time_gpu, history_cpu, history_gpu)
    """
    logger.info("--- PyTorch Logistic Regression ---")

    # Convert NumPy arrays to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_np.astype(np.float32))
    y_train_tensor = torch.tensor(y_train_np.astype(np.float32))
    X_val_tensor = torch.tensor(X_val_np.astype(np.float32))
    y_val_tensor = torch.tensor(y_val_np.astype(np.float32))
    X_test_tensor = torch.tensor(X_test_np.astype(np.float32))
    y_test_tensor = torch.tensor(y_test_np.astype(np.float32))

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.BCEWithLogitsLoss()

    all_metrics_cpu = None
    all_metrics_gpu = None
    time_cpu_val = float('nan')
    time_gpu_val = float('nan')
    history_cpu = None
    history_gpu = None

    # Training on CPU
    logger.info("Training on CPU...")
    device_cpu = torch.device("cpu")
    model_cpu = PyTorchLogisticRegression(input_dim)
    optimizer_cpu = optim.SGD(model_cpu.parameters(), lr=lr)

    start_time_cpu = time.time()
    history_cpu = train_pytorch_model(model_cpu, train_loader, val_loader, criterion, optimizer_cpu, device_cpu,
                                      epochs=epochs,
                                      verbose=False)
    time_cpu_val = time.time() - start_time_cpu
    logger.info(f"CPU Training time: {time_cpu_val:.2f} seconds")

    metrics_train_cpu = evaluate_pytorch_model(model_cpu, train_loader, device_cpu, criterion)
    metrics_val_cpu = evaluate_pytorch_model(model_cpu, val_loader, device_cpu, criterion)
    metrics_test_cpu = evaluate_pytorch_model(model_cpu, test_loader, device_cpu, criterion)
    all_metrics_cpu = {'train': metrics_train_cpu, 'val': metrics_val_cpu, 'test': metrics_test_cpu}
    logger.info(
        f"PyTorch CPU - Test Accuracy: {metrics_test_cpu['Accuracy']:.4f}, F1: {metrics_test_cpu['F1']:.4f}, ROC_AUC: {metrics_test_cpu['ROC_AUC'] if isinstance(metrics_test_cpu['ROC_AUC'], str) else metrics_test_cpu['ROC_AUC']:.4f}")

    # Training on GPU (if available)
    if torch.cuda.is_available():
        logger.info("Training on GPU...")
        device_gpu = torch.device("cuda")
        model_gpu = PyTorchLogisticRegression(input_dim)  # Re-initialize model for GPU
        optimizer_gpu = optim.SGD(model_gpu.parameters(), lr=lr)

        start_time_gpu = time.time()
        history_gpu = train_pytorch_model(model_gpu, train_loader, val_loader, criterion, optimizer_gpu, device_gpu,
                                          epochs=epochs,
                                          verbose=False)
        time_gpu_val = time.time() - start_time_gpu
        logger.info(f"GPU Training time: {time_gpu_val:.2f} seconds")

        metrics_train_gpu = evaluate_pytorch_model(model_gpu, train_loader, device_gpu, criterion)
        metrics_val_gpu = evaluate_pytorch_model(model_gpu, val_loader, device_gpu, criterion)
        metrics_test_gpu = evaluate_pytorch_model(model_gpu, test_loader, device_gpu, criterion)
        all_metrics_gpu = {'train': metrics_train_gpu, 'val': metrics_val_gpu, 'test': metrics_test_gpu}
        logger.info(
            f"PyTorch GPU - Test Accuracy: {metrics_test_gpu['Accuracy']:.4f}, F1: {metrics_test_gpu['F1']:.4f}, ROC_AUC: {metrics_test_gpu['ROC_AUC'] if isinstance(metrics_test_gpu['ROC_AUC'], str) else metrics_test_gpu['ROC_AUC']:.4f}")
    else:
        logger.warning("GPU not available. Skipping GPU training.")

    return all_metrics_cpu, all_metrics_gpu, time_cpu_val, time_gpu_val, history_cpu, history_gpu
