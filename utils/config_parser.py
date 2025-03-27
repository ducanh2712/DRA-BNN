import xml.etree.ElementTree as ET
import torch

def parse_config(xml_file):
    """Parse configuration from an XML file and return a dictionary."""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    config = {}
    
    # Training parameters
    training = root.find('training')
    config['num_epochs'] = int(training.find('num_epochs').text)
    config['min_lr'] = float(training.find('min_lr').text)
    config['patience'] = int(training.find('patience').text)
    config['early_stopping_delta'] = float(training.find('early_stopping_delta').text)
    config['batch_size'] = int(training.find('batch_size').text)
    config['num_classes'] = int(training.find('num_classes').text)
    
    # Paths
    paths = root.find('paths')
    config['data_dir'] = paths.find('data_dir').text
    config['log_dir'] = paths.find('log_dir').text
    
    # Device
    device_type = root.find('device/type').text
    if device_type == 'auto':
        config['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_type == 'cuda':
        config['device'] = torch.device("cuda")
    elif device_type == 'cpu':
        config['device'] = torch.device("cpu")
    else:
        raise ValueError(f"Unsupported device type in config: {device_type}")
    
    return config