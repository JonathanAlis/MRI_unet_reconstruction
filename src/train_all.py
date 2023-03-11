from trainer import Trainer
from datasets import OriginalReconstructionDataset
from device import get_default_device, to_device, DeviceDataLoader
from sklearn.model_selection import ParameterGrid
import torch


def train_all(num_radial_lines, exp_params, device='cuda:0', verbose=False):
    experiment_name=f"MRIREC_{num_radial_lines}"
    run_params = {"description":f"Reconstruction using {num_radial_lines} radial lines",
              "tags":{'release.version':'2.0.0'}}
    grid_exp = ParameterGrid(exp_params)

    for p_model in grid_exp:
    
        print('Training for:')
        print({'model': p_model['model'], 'rectype': p_model['rectype'], 'max_epochs': p_model['max_epochs'], 'learning_rate': p_model['learning_rate'], 'batch_size': p_model['batch_size']})

        model_name = p_model['model']
        rectype = p_model['rectype']
        lr_type, learning_rate = p_model['learning_rate']
        model_trainer=Trainer(model_name, rectype, num_radial_lines, models_folder='MRIrec_experiments',
                              device=device, lr_type=lr_type, learning_rate=learning_rate, verbose=verbose)
        epochs = p_model['max_epochs']
        
        batch_size = p_model['batch_size']
        if batch_size=='auto':
            batch_size=model_trainer.auto_find_batch_size()

        train_ds= OriginalReconstructionDataset(num_radial_lines, rectype, set='train')
        train_dl= DeviceDataLoader(torch.utils.data.DataLoader(train_ds, batch_size=batch_size), device)
        val_ds=OriginalReconstructionDataset(num_radial_lines, rectype, set='val')
        val_dl= DeviceDataLoader(torch.utils.data.DataLoader(val_ds, batch_size=batch_size), device)

        #train loop:
        history=model_trainer.train(train_dl, val_dl, max_epochs=epochs)



exp_params={"model": ['Unet','ResnetUnet', 'ConvUNeXt'],
            "rectype": OriginalReconstructionDataset.rectypes_list,
            "learning_rate":[('constant',1e-3)],#[('step10',1e-3)],#('constant', 1e-5)],#[1e-4],#, 'exp', 'plateau'],
            "max_epochs":[100],
            "batch_size": ['auto'],
}


radial_lines=[20,40,60,80,100]

for rl in OriginalReconstructionDataset.all_radial_lines:
    train_all(rl, exp_params, device='cuda:0', verbose=True)

        