
import pickle
import os

def data_initialization(file_path,num_of_epcho):
    data = {'train':{f'epoch_{i + 1}': False for i in range(num_of_epcho)},
            'test': {'1':False}}

    if not os.path.exists(file_path):
        os.makedirs(file_path)
    with open(os.path.join(file_path,'experiments_data.pkl'), 'wb') as file:
        pickle.dump(data, file)

def update_experiment_data(file_path, epoch, src_tensor,tgt_tensor,train=True):
    # Load the existing data
    with open(os.path.join(file_path,'experiments_data.pkl'), 'rb') as file:
        data = pickle.load(file)
        # Update the specific epoch of the experiment
    data['train' if train else 'test'][str(epoch)] = {'src':src_tensor,
                               'tgt':tgt_tensor}
    # Save the updated data back to the file
    with open(os.path.join(file_path,'experiments_data.pkl'), 'wb') as file:
        pickle.dump(data, file)

