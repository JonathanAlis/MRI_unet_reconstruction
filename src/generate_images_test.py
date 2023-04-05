import os
import pandas as pd
from torchvision.utils import save_image
from trainer import Trainer
from datasets import OriginalReconstructionDataset
from device import get_default_device, to_device, DeviceDataLoader
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid
import torch
import skimage
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import pandas as pd
lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True)

def batch_metrics(orig, result):
    #print(torch.cat([orig, orig, orig], dim=1).shape)
    #print(torch.cat([result, result, result], dim=1).shape)
    lpips_batch=lpips.net(torch.cat([orig, orig, orig], dim=1),
                                  torch.cat([result, result, result], dim=1)
                                  ).squeeze()
    lpips_batch=list(lpips_batch.detach().numpy())

    ssim_batch=[]
    psnr_batch=[]
    for i in range(orig.shape[0]):
        ssim_batch.append(skimage.metrics.structural_similarity(orig[i,0,...].detach().numpy(),result[i,0,...].detach().numpy()))
        psnr_batch.append(skimage.metrics.peak_signal_noise_ratio(orig[i,0,...].detach().numpy(),result[i,0,...].detach().numpy()))
    return psnr_batch, ssim_batch, lpips_batch

def inference(model, MRIdataloader):
    model.eval()
    with torch.no_grad(): # No need to track the gradients
        for image_batch, image_noisy, img_idx in tqdm(MRIdataloader):
            result = model(image_noisy)
            yield result, image_batch, image_noisy, img_idx

def test_all(num_radial_lines, exp_params, device='cuda:0', verbose=False):
    grid_exp = ParameterGrid(exp_params)

    for p_model in grid_exp:
    
        print('Testing for:')
        print({'model': p_model['model'], 'rectype': p_model['rectype'], 'max_epochs': p_model['max_epochs'], 'learning_rate': p_model['learning_rate'], 'batch_size': p_model['batch_size']})

        model_name = p_model['model']
        rectype = p_model['rectype']
        lr_type, learning_rate = p_model['learning_rate']
        model_trainer=Trainer(model_name, rectype, num_radial_lines, models_folder='MRIrec_experiments', last_or_best='best',
                              device=device, lr_type=lr_type, learning_rate=learning_rate, verbose=verbose)
        epochs = p_model['max_epochs']
        
        batch_size = p_model['batch_size']
        if batch_size=='auto':
            batch_size=model_trainer.auto_find_batch_size()

        test_ds= OriginalReconstructionDataset(num_radial_lines, rectype, set='test', return_idx=True)
        test_dl= DeviceDataLoader(torch.utils.data.DataLoader(test_ds, batch_size=batch_size), device)

        input_psnr=[]
        input_ssim=[]
        input_lpips=[]
        output_psnr=[]
        output_ssim=[]
        output_lpips=[]
        imgs_ids=[]
        save_path=f"result_images/{p_model['model']}_{num_radial_lines}lines_{p_model['rectype']}_{model_trainer.best_epoch}epochs/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        
        for output_batch, original_batch, input_batch, image_idx in inference(model_trainer.model, test_dl):
            for i, idx in enumerate(list(image_idx)):
                img_id=test_ds.images[idx].replace('_normalized.png','')
                save_image(output_batch[i,...], f'{save_path}{img_id}.png')
                imgs_ids.append(img_id)
            #p0,s0,l0=batch_metrics(original_batch.cpu(), input_batch.cpu())
            p1,s1,l1=batch_metrics(original_batch.cpu(), output_batch.cpu())
            #input_psnr+=p0
            #input_ssim+=s0
            #input_lpips+=l0
            output_psnr+=p1
            output_ssim+=s1
            output_lpips+=l1
            #print(original_batch.shape, input_batch.shape, output_batch.shape)
        #print(input_psnr)
        df=pd.DataFrame(list(zip(imgs_ids, output_psnr, output_ssim, output_lpips)), columns =['id', 'psnr', 'ssim', 'lpips'])
        print(df)
        df.to_csv(f'{save_path}metrics.csv')
        #print(recs.shape)
        #train loop:
        #history=model_trainer.train(train_dl, val_dl, max_epochs=epochs)
        #del model_trainer
        #torch.cuda.empty_cache()



exp_params={"model": ['Unet','ResnetUnet', 'ConvUNeXt'],
            "rectype": OriginalReconstructionDataset.rectypes_list,
            "learning_rate":[('constant',1e-3)],#[('step10',1e-3)],#('constant', 1e-5)],#[1e-4],#, 'exp', 'plateau'],
            "max_epochs":[100],
            "batch_size": ['auto'],
}


radial_lines=[20,40,60,80,100]# 
 
for rl in radial_lines:#OriginalReconstructionDataset.all_radial_lines:
    test_all(rl, exp_params, device='cuda:0', verbose=True)

        