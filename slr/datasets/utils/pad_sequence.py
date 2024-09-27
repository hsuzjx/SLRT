import torch


def pad_video_sequence(sequences, batch_first=False, padding_value=0.0, left_pad_length=6, right_pad_length=6):
    """
    Pad the first dimension of each element in a list of video sequences while keeping other dimensions unchanged.

    Args:
    - sequences: A list of video sequences, where each sequence is a Tensor.
    - batch_first: Whether to put the batch dimension first (default: False, output shape is [T, B, ...]).
    - padding_value: Value used for padding (default: 0.0).
    - left_pad_length: Length of padding on the left side (default: 6).
    - right_pad_length: Length of padding on the right side (default: 6).

    Returns:
    - A Tensor with all sequences padded to the same length in the first dimension.
    """
    # Handle empty list
    if not sequences:
        return torch.tensor([], dtype=torch.float32)

    # Get the length of each sequence
    lengths = [seq.shape[0] for seq in sequences]  # Assuming the length dimension is the first one (T)
    max_length = max(lengths)
    batch_size = len(sequences)

    # Get the size of other dimensions
    other_dims = sequences[0].shape[1:]  # Assuming other dimensions start from the second one

    # Create padded tensor
    batch = torch.full(
        (left_pad_length + max_length + right_pad_length, batch_size, *other_dims),
        padding_value, dtype=sequences[0].dtype
    )

    # Fill in the data
    for i, seq in enumerate(sequences):
        start_index = left_pad_length
        end_index = start_index + lengths[i]
        batch[start_index:end_index, i] = seq

    # If batch_first is True, swap the dimensions
    if batch_first:
        batch = batch.transpose(0, 1)  # Swap dimensions to (B, T, ...)

    return batch


def pad_label_sequence(sequences, batch_first=False, padding_value=0.0):
    """
    Pad the first dimension of each element in a list of label sequences while keeping other dimensions unchanged.

    Args:
    - sequences: A list of label sequences, where each sequence is a Tensor.
    - batch_first: Whether to put the batch dimension first (default: False, output shape is [T, B]).
    - padding_value: Value used for padding (default: 0.0).

    Returns:
    - A Tensor with all sequences padded to the same length in the first dimension.
    """
    # Handle empty list
    if not sequences:
        return torch.tensor([], dtype=torch.float32)

    # Get the length of each sequence
    lengths = [seq.shape[0] for seq in sequences]  # Assuming the length dimension is the first one (T)
    max_length = max(lengths)
    batch_size = len(sequences)

    # Create padded tensor
    batch = torch.full((max_length, batch_size), padding_value, dtype=sequences[0].dtype)

    # Fill in the data
    for i, seq in enumerate(sequences):
        batch[:len(seq), i] = seq

    # If batch_first is True, swap the dimensions
    if batch_first:
        batch = batch.transpose(0, 1)  # Swap dimensions to (B, T)

    return batch
