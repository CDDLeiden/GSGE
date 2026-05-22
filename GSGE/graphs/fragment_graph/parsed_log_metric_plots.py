import re
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def parse_log_file(log_text):
    """
    Parse the log text to extract epoch, phase, and metrics.
    Returns a dictionary with metrics over epochs for Train and Val.
    """
    data = defaultdict(lambda: defaultdict(list))
    
    epoch_pattern = re.compile(r"Epoch \[(\d+)/(\d+)\] ------------------------------------- (Train|Val)", re.IGNORECASE)
    metric_pattern = re.compile(
        r"(Train|Val)\s+(Atom Loss|Edge Loss|Total Loss|atom_accuracy|atom_f1|balanced_atom_acc|atom_num_r2|"
        r"edge_type_accuracy|edge_type_f1|balanced_edge_type_acc|edge_num_r2|adj_accuracy|adj_f1|balanced_adj_acc)"
        r"\s*:\s*([\d.]+)", re.IGNORECASE
    )
    
    current_epoch = None
    for line in log_text.split('\n'):
        epoch_match = re.match(epoch_pattern, line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            total_epochs = int(epoch_match.group(2))
            current_phase = epoch_match.group(3)
            continue
        
        metric_match = re.findall(metric_pattern, line)
        if metric_match and current_epoch is not None:
            for phase, metric_name, value in metric_match:
                value = float(value)
                data[current_epoch][f"{phase}_{metric_name}"].append(value)
    
    return data

def plot_metrics(data):
    """
    Generate plots for all metrics over epochs, comparing Train and Val.
    """
    epochs = sorted(data.keys())
    metric_groups = {
        "Loss": ["Atom Loss", "Edge Loss", "Total Loss"],
        "Accuracy": ["atom_accuracy", "edge_type_accuracy", "adj_accuracy"],
        "F1 Score": ["atom_f1", "edge_type_f1", "adj_f1"],
        "Balanced Accuracy": ["balanced_atom_acc", "balanced_edge_type_acc", "balanced_adj_acc"],
        "R² Score": ["atom_num_r2", "edge_num_r2"]
    }
    
    for group_name, metrics in metric_groups.items():
        plt.figure(figsize=(10, 6))
        has_data = False
        for metric in metrics:
            train_key = f"Train_{metric}"
            val_key = f"Val_{metric}"
            train_values = []
            val_values = []
            
            for e in epochs:
                train_val = data[e].get(train_key, [np.nan])[0]
                val_val = data[e].get(val_key, [np.nan])[0]
                train_values.append(train_val)
                val_values.append(val_val)
            
            if not all(np.isnan(train_values)) or not all(np.isnan(val_values)):
                has_data = True
                plt.plot(epochs, train_values, label=f"Train {metric}", marker='o')
                plt.plot(epochs, val_values, label=f"Val {metric}", marker='x')
        
        if has_data:
            plt.title(f"{group_name} Over Epochs")
            plt.xlabel("Epoch")
            plt.ylabel(group_name)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        else:
            print(f"No data to plot for {group_name}")
            plt.close()

if __name__ == '__main__':
    # Get Path of the GSGE package
    from pathlib import Path
    gsge_root = Path(__file__).resolve().parents[1]

    #example use to plot in ipynb
    from GSGE import get_use_examples_dir
    examples_dir = get_use_examples_dir()
    if examples_dir is None:
        raise RuntimeError("Cannot find use_examples directory.")
    log_path = examples_dir / '03_GAE' / 'v2' / 'train_GAE_v5_vocab.log'
    with open(log_path, 'r') as f:
        log_text = f.read()

    """" log_text ='
    Starting training from scratch
    Epoch [1/300] ------------------------------------- Train
    Train Atom Loss: 167.0284, Train Edge Loss: 134.5412, Train Total Loss: 301.5696
    Train atom_accuracy: 0.5809, Train atom_f1: 0.0834, Train balanced_atom_acc: 0.0930, Train atom_num_r2: 0.7412
    Train edge_type_accuracy: 0.7784, Train edge_type_f1: 0.3299, Train balanced_edge_type_acc: 0.3934, Train edge_num_r2: 0.7972
    Train adj_accuracy: 0.8368, Train adj_f1: 0.7680, Train balanced_adj_acc: 0.8119

    Epoch [1/300] ------------------------------------- Val
    Val Atom Loss: 36.0419, Val Edge Loss: 26.9759, Val Total Loss: 63.0177
    Val atom_accuracy: 0.6203, Val atom_f1: 0.1216, Val balanced_atom_acc: 0.1281, Val atom_num_r2: 0.9916
    Val edge_type_accuracy: 0.8312, Val edge_type_f1: 0.4967, Val balanced_edge_type_acc: 0.4827, Val edge_num_r2: 0.9829
    Val adj_accuracy: 0.8474, Val adj_f1: 0.7823, Val balanced_adj_acc: 0.8274'
    """

    # Parse log text and plot
    data = parse_log_file(log_text)
    plot_metrics(data)
