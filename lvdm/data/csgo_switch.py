import torch
from lvdm.data.csgovid import CSGOVID
import random
import os

class CSGOVIDSwitchWeaponDataset(CSGOVID):
    """
    Extension of CSGOVID dataset that modifies actions to simulate weapon switching
    with multiple variations for the same video clip.
    """
    def __init__(self, num_variations=5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_variations = num_variations
        # We expand the dataset size by num_variations
        
    def __len__(self):
        return len(self.file_list) * self.num_variations
    
    def __getitem__(self, index):
        # Map extended index back to original file index
        original_index = index // self.num_variations
        variation_idx = index % self.num_variations
        
        # Get original data
        data = super().__getitem__(original_index)
        
        # Modify action
        actions_tensor = data['action'] # [T, 51]
        T = actions_tensor.shape[0]
        
        # Create switch weapon action
        # We keep the history actions (before cond_frame) intact if possible,
        # but here we might want to override from a certain point.
        # Assuming we want to enforce switch weapon behavior for the whole "future" part.
        # However, the model uses cond_frame (e.g. 4 or 9) as history.
        # Let's modify actions starting from cond_frame.
        
        start_mod_idx = self.cond_frame if hasattr(self, 'cond_frame') else 0
        
        # Strategy: 
        # 1. First 5 frames: Press '1' (Primary Weapon) to ensure/force a known state.
        # 2. Remaining frames: Press '3' (Knife) which has a very distinct animation.
        # This maximizes the chance of seeing a visible switch action regardless of initial state.
        
        N_KEYS = 11
        N_MOUSE_X = 23
        N_MOUSE_Y = 15
        
        for t in range(start_mod_idx, T):
            # Reset action at t
            # Keys: 11 dims
            keys = torch.zeros(N_KEYS)
            
            # Determine key press based on time relative to start_mod_idx
            rel_t = t - start_mod_idx
            
            # Pulse trigger: press only for one frame
            if rel_t == 0:
                keys[7] = 1.0 # Press '1' (Primary) once at start
            elif rel_t == 5:
                keys[9] = 1.0 # Press '3' (Knife) once at frame 5
            # Else: keys remain all zeros (released)
            
            # Mouse clicks: 0
            l_click = torch.zeros(1)
            r_click = torch.zeros(1)
            
            # Mouse movement: center
            mouse_x_idx = 11 # Center
            mouse_y_idx = 7  # Center
            
            mouse_x = torch.zeros(N_MOUSE_X)
            mouse_x[mouse_x_idx] = 1.0
            
            mouse_y = torch.zeros(N_MOUSE_Y)
            mouse_y[mouse_y_idx] = 1.0
            
            # Reassemble
            actions_tensor[t] = torch.cat([keys, l_click, r_click, mouse_x, mouse_y])
            
        data['action'] = actions_tensor
        
        # Update path to avoid overwriting files if logger uses it for filename
        # Appending variation index to path stem
        original_path = data['path']
        base, ext = os.path.splitext(original_path)
        data['path'] = f"{base}_var{variation_idx}{ext}" # we do nothing here, just pure repitition for variation_idx times, as monte carlo sampling
        
        return data

