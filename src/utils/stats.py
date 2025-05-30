def calculate_seizure_event_durations(df, label_col='label', segment_duration_seconds=12):
    """
    Calculates the duration of contiguous seizure events from a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with a label column indicating seizure (1) or non-seizure (0).
                           It's assumed segments are sequential.
        label_col (str): Name of the column containing seizure labels (0 or 1).
        segment_duration_seconds (int): Duration of each segment in seconds.

    Returns:
        list: A list of seizure event durations in seconds.
    """
    if label_col not in df.columns:
        print(f"Warning: Label column '{label_col}' not found. Returning empty list for durations.")
        return []
    if df.empty:
        return []

    labels = df[label_col]
    
    # Identify blocks of consecutive identical labels
    # A new block starts when the label value changes from the previous one
    block_ids = (labels != labels.shift()).cumsum()
    
    # Group by these block_ids and also by the label value itself
    # Then, count the number of segments (size) in each block
    block_sizes = df.groupby([block_ids, labels]).size()
    
    # Filter for blocks where the label indicates a seizure (label == 1)
    # The index of block_sizes is a MultiIndex (block_id, label_value)
    # We select where the label_value part of the index is 1
    try:
        # Access elements where the second level of the MultiIndex is 1
        seizure_event_lengths_in_segments = block_sizes.xs(1, level=1)
    except KeyError:
        # This means there were no blocks with label 1 (no seizures)
        return []
        
    if seizure_event_lengths_in_segments.empty:
        return []
        
    # Convert segment counts to durations in seconds
    seizure_durations_seconds = [length * segment_duration_seconds for length in seizure_event_lengths_in_segments]
    
    return seizure_durations_seconds