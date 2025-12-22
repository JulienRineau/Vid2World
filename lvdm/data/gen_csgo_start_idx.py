import numpy as np
import random

def generate_start_indices(num_samples=500, max_start=1000-16, seed=42):
    """
    Generate random start indices for CSGO validation set.
    
    Args:
        num_samples: Number of start indices to generate
        max_start: Maximum start index (1000-16 for CSGO dataset)
        seed: Random seed for reproducibility
    """
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    # Generate random start indices
    start_indices = np.random.randint(0, max_start, size=num_samples)
    
    # Save to file
    save_path = "csgo_validation_start_indices.npy"
    np.save(save_path, start_indices)
    print(f"Generated {num_samples} start indices and saved to {save_path}")
    print(f"Start indices range: [{start_indices.min()}, {start_indices.max()}]")
    
    return start_indices

if __name__ == "__main__":
    start_indices = generate_start_indices()
