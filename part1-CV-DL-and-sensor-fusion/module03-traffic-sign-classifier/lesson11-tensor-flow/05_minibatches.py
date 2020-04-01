import math
def batches(batch_size, features, labels):
    """
    Create batches of features and labels
    :param batch_size: The batch size
    :param features: List of features
    :param labels: List of labels
    :return: Batches of (Features, Labels)
    """
    # Store number of features for optimization
    n_inputs = len(features)
    
    # Check that the number of inputs is the same than the number of labels
    assert n_inputs == len(labels)
    
    # DONE: Implement batching
    # List of batches
    batches = []
    
    # Jump trough data using bath-sized steps
    for idx in range(0, n_inputs, batch_size):
        # Compute ending index for this batch
        end_idx = idx+batch_size
        
        # If there are enough samples to build a full mini-batch
        if end_idx<n_inputs:
            batch = [features[idx:end_idx], labels[idx:end_idx]]
        # If not, get the remaining samples
        else:
            batch = [features[idx:], labels[idx:]]
        
        # Append to batches list
        batches.append(batch)

    return batches

