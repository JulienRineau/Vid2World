import numpy as np
import os
from tqdm import tqdm
import argparse
import tensorflow_datasets as tfds

DISPLAY_KEY = {
    'taco_play': 'rgb_static',
    'roboturk': 'front_rgb',
    'viola': 'agentview_rgb',
    'language_table': 'rgb',
    'stanford_robocook_converted_externally_to_rlds1': 'image_1',
    'stanford_robocook_converted_externally_to_rlds2': 'image_2',
    'stanford_robocook_converted_externally_to_rlds3': 'image_3',
    'stanford_robocook_converted_externally_to_rlds4': 'image_4',
    'uiuc_d3field1': 'image_1',
    'uiuc_d3field2': 'image_2',
    'uiuc_d3field3': 'image_3',
    'uiuc_d3field4': 'image_4',
}

def get_dataset_path(parent_dir, dataset_name):
    if dataset_name == 'robo_net' or dataset_name == 'cmu_playing_with_food':
        version = '1.0.0'
    elif dataset_name == 'language_table':
        version = '0.0.1'
    elif dataset_name[:-1] == 'uiuc_d3field' or dataset_name[:-1] == 'stanford_robocook_converted_externally_to_rlds':
        dataset_name = dataset_name[:-1]
        version = '0.1.0'
    else:
        version = '0.1.0'
    return os.path.join(parent_dir, dataset_name, version)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='fractal20220817_data')
    parser.add_argument('--input_path', type=str, default='/data2/tensorflow_datasets')
    parser.add_argument('--output_path', type=str, default='./frac_info_temp')
    parser.add_argument('--max_num_episodes', default=None, type=int)
    args = parser.parse_args()

    dataset_name = args.dataset_name
    ds = tfds.builder_from_directory(builder_dir=get_dataset_path(args.input_path, dataset_name)).as_dataset()
    display_key = DISPLAY_KEY.get(dataset_name, 'image')
    root_path = os.path.join(args.output_path, dataset_name)
    os.makedirs(root_path, exist_ok=True)

    num_episodes = 0
    for key in ds:
        bar = tqdm(enumerate(ds[key]))
        for i, episode in bar:
            if os.path.exists(os.path.join(root_path, f'{key}_eps_{i:08d}.npz')):
                continue
            
            # Extract frames using display_key
            frames = np.array([step['observation'][display_key] for step in episode['steps']])
            
            # Extract and process actions
            action = [step['action'] for step in episode['steps']]
            action = [np.concatenate([
                action[i]['base_displacement_vector'],
                action[i]['base_displacement_vertical_rotation'],
                action[i]['gripper_closedness_action'],
                action[i]['rotation_delta'],
                action[i]['terminate_episode'],
                action[i]['world_vector']
            ]) for i in range(len(action))]
            action = np.array(action)
            
            bar.set_postfix(epslen=len(frames))
            
            # Save both image and action to npz file
            # RT-1 dataloader expects: 'image' [T, H, W, C] and 'action' [T, M]
            np.savez_compressed(
                os.path.join(root_path, f'{key}_eps_{i:08d}.npz'),
                image=frames,  # [T, H, W, C]
                action=action  # [T, M]
            )
            
            num_episodes += 1
            if args.max_num_episodes is not None and num_episodes >= args.max_num_episodes:
                break
