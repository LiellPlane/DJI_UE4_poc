import numpy as np





def decode_barcode(data, threshold=0.5):
    """
    Decodes a barcode from a 1D uint8 array by detecting transitions.
    
    Parameters:
    - data: np.ndarray, 1D array of uint8 values representing the scanned barcode.
    - threshold: Float between 0 and 1; intensity value used to binarize the normalized data.
    
    Returns:
    - transitions: List of indices where transitions occur.
    - widths: List of widths of bars and spaces.
    - binary_data: Binarized version of the input data after normalization.
    """
    # Step 1: Normalize the data to range [0, 1]
    data_min = data.min()
    data_max = data.max()
    if data_max > data_min:
        normalized_data = (data - data_min) / (data_max - data_min)
    else:
        normalized_data = np.zeros_like(data, dtype=float)
    
    # Step 2: Thresholding using a fixed threshold (e.g., 0.5)
    binary_data = (normalized_data > threshold).astype(int)
    
    # Step 3: Finding Transitions
    diff_data = np.diff(binary_data)
    transition_indices = np.flatnonzero(diff_data) + 1  # Add 1 due to diff shift
    transitions = transition_indices.tolist()
    
    # Include start and end positions
    positions = [0] + transitions + [len(data)]
    
    # Step 4: Extracting Bar Widths
    widths = np.diff(positions)
    
    return transitions, widths.tolist(), binary_data


def main():
    
if __name__ == "__main__":
    main()