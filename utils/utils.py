import os
import shutil
import glob
import torch

def get_most_recent_checkpoint(checkpoint_dir, option=None):
    
    if option is None:
        checkpoint_paths = [path for path in glob.glob(f"{checkpoint_dir}/checkpoint_*")]
    else:
        checkpoint_paths = [path for path in glob.glob(f"{checkpoint_dir}/checkpoint_{option}_*")]
        
    lastest_checkpoint=None

    if len(checkpoint_paths) != 0:
        idxes = [int(os.path.basename(path).split('_')[-1]) for path in checkpoint_paths] # [scalar]
        max_idx = max(idxes) # scalar
        if option is None:
            lastest_checkpoint = os.path.join(checkpoint_dir, f"checkpoint_{max_idx}")        
        else:
            lastest_checkpoint = os.path.join(checkpoint_dir, f"checkpoint_{option}_{max_idx}")        

        print(f" [*] Found lastest checkpoint: {lastest_checkpoint}")
    else:
        None
        
    return lastest_checkpoint

def save_checkpoint(model, optimizer, iteration, num_gpus, filepath, option=None):

    if option is None:
        save_path = f'{filepath}/checkpoint_{iteration}'
    else:
        save_path = f'{filepath}/checkpoint_{option}_{iteration}'
    print(f"Saving model and optimizer state at iteration {iteration} to {save_path}")

    torch.save({'iteration': iteration,
                'model': (model.module if num_gpus > 1 else model).state_dict(),
                'optimizer': optimizer.state_dict()},
                save_path)

    return True

def load_checkpoint(hparams, model, optimizer, device, option=None):
    
    checkpoint_path = f"{hparams.output_directory}/{hparams.log_directory}"
    checkpoint_path = get_most_recent_checkpoint(checkpoint_path, option)
    iteration = -1
          
    if checkpoint_path is not None:        
        checkpoint_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint_dict['model'], strict=False)  
        optimizer.load_state_dict(checkpoint_dict['optimizer']) 
        iteration = checkpoint_dict['iteration'] 
        
        print(f'Loading: {checkpoint_path} // iteration: {iteration} ---------------------------------------')

    return model, optimizer, iteration

def display_result_and_save_tensorboard(writer, result_dict, iteration):

    for key, value in result_dict.items():        
        writer.add_scalar(key, value, iteration)

    return writer

def copy_code(output_directory, log_directory): 

    save_path=f'{output_directory}/{log_directory}/code/'

    npz_path = list()
    for folder in ['model', 'text', 'utils', 'vocoder', '']:

        path_list1 = glob.glob(os.path.join(folder, '*.py'))
        path_list = [path.split('/')[-1] for path in path_list1] # file name
        if not os.path.exists(os.path.join(save_path, folder)):
            os.makedirs(os.path.join(save_path, folder))

        if folder == "text": # 하위폴더 추가
            path_list2 = glob.glob(os.path.join(folder, '*/*.py'))
            path_list += ['/'.join(path.split('/')[-2:]) for path in path_list2] # file name
            folders = set([path.split('/')[-2] for path in path_list])
            for f2 in folders:
                if not os.path.exists(os.path.join(save_path, folder, f2)):
                    os.makedirs(os.path.join(save_path, folder, f2))

        npz_path += [(os.path.join(folder, target), os.path.join(save_path, folder, target)) for target in path_list]
        
    for source_list, target_list in npz_path:
        shutil.copy(source_list, target_list)

    return True

