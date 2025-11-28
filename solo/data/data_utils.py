import torch

def time_to_frequency(data):
    """
    Convert time series data to frequency domain using PyTorch.
    Args:
    data (torch.Tensor): The time series data, expected shape (batch_size, sequence_length)
    Returns:
    torch.Tensor: The frequency domain representation of the data.
    """
    # Assuming data is a 2D tensor where each row represents a time series sample
    frequency_data = torch.fft.fft(data)
    return frequency_data