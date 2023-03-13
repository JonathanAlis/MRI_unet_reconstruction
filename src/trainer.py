from tqdm import tqdm
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import models
import os
import time

class Trainer():
    def __init__(self, model_name, rectype, radial_lines, models_folder='MRIrec_experiments', 
                 device='cpu',lr_type='constant', learning_rate=1e-3, batch_size='auto', verbose=False) -> None:
        self.model_name=model_name
        self.rectype=rectype
        self.radial_lines=radial_lines
        self.lr_type=lr_type
        self.learning_rate=learning_rate
        self.device=device
        self.batch_size=batch_size
        self.models_folder=models_folder
        self.verbose=verbose

        self.model=self.select_model()        
        self.get_saved_history()   
        self.load_model()
    def number_of_channels(self):
        return len(self.rectype.split('_'))
    
    def select_model(self):
        if self.model_name=='Unet':
            model=models.Unet(num_inputs=self.number_of_channels()) #1 a 3 canais
        elif self.model_name=='ResnetUnet':        
            model = models.ResnetUnet(in_channels=self.number_of_channels())
        elif self.model_name=='ConvUNeXt':
            model = models.ConvUNeXt(in_channels=self.number_of_channels(),num_classes=1)
            #IF TO ADD NEW MODEL IMPLEMENT H#RE
        else:
            raise Exception('Model not found...')
        return model

    def define_optimizer(self, params_to_optimize,max_epochs=100):
        if self.lr_type == 'constant':
            self.optimizer = torch.optim.Adam(params_to_optimize, lr=self.learning_rate, weight_decay=1e-05)
            self.lr_scheduler = torch.optim.lr_scheduler.ConstantLR(self.optimizer, factor=1)
        if self.lr_type == 'step10':
            self.optimizer = torch.optim.Adam(params_to_optimize, lr=self.learning_rate, weight_decay=1e-05)
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)        
        if self.lr_type =='exp':
            self.optimizer  = torch.optim.Adam(params_to_optimize, lr=self.learning_rate, weight_decay=1e-05)
            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.000001**(1/max_epochs), last_epoch=- 1, verbose=False)
        elif self.lr_type == 'plateau':
            self.optimizer = torch.optim.Adam(params_to_optimize, lr=self.learning_rate, weight_decay=1e-05)
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1**(1/2), patience=10, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)
        else:
            self.optimizer = torch.optim.Adam(params_to_optimize, lr=self.learning_rate, weight_decay=1e-05)
            self.lr_scheduler = torch.optim.lr_scheduler.ConstantLR(self.optimizer, factor=1)


    def load_model(self):        
        if self.verbose:
            print('Loading model')
        params_to_optimize = [{'params': self.model.parameters()}]        
        self.define_optimizer(params_to_optimize)           
        try:
            self.checkpoint = torch.load(self.last_checkpoint)
            self.model.load_state_dict(self.checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
            self.lr_scheduler.load_state_dict(self.checkpoint['scheduler_state_dict'])
            self.epoch = self.checkpoint['epoch']
            self.train_loss = self.checkpoint['train_loss']
            self.val_loss = self.checkpoint['val_loss']
        except:
            print('No saved model found or not able to load.')
            self.epoch = 0
            self.train_loss=float('inf')
            self.val_loss=float('inf')            
        self.model.to(self.device)
        self.optimizer_to_device()


    def test_scheduler(self, max_epochs=30):
        import copy
        llr=[]
        sc=copy.deepcopy(self.lr_scheduler)
        op=copy.deepcopy(self.optimizer)
        for _ in range(max_epochs):
            llr.append(op.param_groups[0]['lr'])
            sc.step()


        
    def save_model(self, path, epoch, train_loss, val_loss):
        torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.lr_scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                }, path)
                
    def optimizer_to_device(self):
        for param in self.optimizer.state.values():
            # Not sure there are any global tensors in the state dict
            if isinstance(param, torch.Tensor):
                param.data = param.data.to(self.device)
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(self.device)
            elif isinstance(param, dict):
                for subparam in param.values():
                    if isinstance(subparam, torch.Tensor):
                        subparam.data = subparam.data.to(self.device)
                        if subparam._grad is not None:
                            subparam._grad.data = subparam._grad.data.to(self.device)

    def get_saved_history(self):  
        self.saved_models_path=f'{self.models_folder}/{self.model_name}_{self.radial_lines}lines_{self.rectype}'
        if not os.path.exists(self.saved_models_path):
            if self.verbose:
                print('Creating folder', self.saved_models_path)
            os.makedirs(self.saved_models_path)
        
        self.csv_filename= f'{self.saved_models_path}/loss_history_{self.lr_type}LR_{self.learning_rate}.csv'
        self.load_history()
        
    def load_history(self):
        if self.verbose:
            print(f'Loading history from {self.csv_filename}')
        if not os.path.exists(self.csv_filename):
            self.df_history = pd.DataFrame(columns = ['epoch','lr_type','learning_rate','train_loss','val_loss', 'checkpoint', 'status', 'time_train', 'time_val'])
            self.last_epoch=0
            self.best_epoch=0
            self.last_checkpoint=''
            self.best_checkpoint=''
            return
         
        folder=self.csv_filename[:self.csv_filename.rfind('/')]
        lr = float(self.csv_filename[self.csv_filename.find('LR_')+3:self.csv_filename.find('.csv')])
        lr_sched=self.csv_filename[self.csv_filename.find('history_')+8:self.csv_filename.find('LR_')]
        assert folder==self.saved_models_path
        assert lr==self.learning_rate
        assert self.lr_type==lr_sched
        #preparing the csv
        self.df_history = pd.read_csv(self.csv_filename)
        self.df_history = self.df_history.loc[:, ~self.df_history.columns.str.contains('^Unnamed')]
        self.df_history = self.df_history.drop_duplicates(subset=['epoch'], keep='first')

        #find best model (lowest validation loss)
        val_sorted_df=self.df_history.sort_values(by=['val_loss'])
        for i, row in val_sorted_df.iterrows():
            if os.path.isfile(row['checkpoint']):
                self.best_epoch=row['epoch']
                self.best_checkpoint=row['checkpoint']
                break
        
        #find last saved model
        epoch_sorted_df=self.df_history.sort_values(by=['epoch'], ascending=False)
        for i, row in epoch_sorted_df.iterrows():
            if os.path.isfile(row['checkpoint']):
                self.last_epoch=row['epoch']
                self.last_checkpoint=row['checkpoint']
                break
        
    def erase_unwanted_models(self):
        #erase rows that do not have a saved model
        df_delete=self.df_history[(self.df_history['epoch']<self.last_epoch) & (self.df_history['epoch']!=self.best_epoch)]
        checkpoints_to_delete=list(df_delete['checkpoint'])
        for f in checkpoints_to_delete:
            try:                                
                os.remove(f)
                if self.verbose:
                     print(f'Deleting file {f}')           
            except:
                pass#if self.verbose:
                    #print(f'couldnt remove {f}')


    ## Training function
    def train_epoch(self, MRIdataloader):
        t_train=time.time()
        # Set train mode
        loss_fn=nn.MSELoss(reduction='sum')
        self.model.train()    
        train_loss = 0
        datasize=0
        # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
        for image_batch, image_noisy in tqdm(MRIdataloader): # with "_" we just ignore the labels (the second element of the dataloader tuple)
            #image_noisy.to(device)
            result = self.model(image_noisy)
            # Evaluate loss        
            loss = loss_fn(result, image_batch)
            # Backward pass        
            self.optimizer.zero_grad()
            loss.backward()
            train_loss+=loss.detach().cpu().numpy()
            datasize+=image_batch.shape[0]
            self.optimizer.step()
        self.lr_scheduler.step()
        train_loss=train_loss if not torch.is_tensor(train_loss) else train_loss.cpu().detach().numpy()
        t_train=time.time()-t_train
        return train_loss/datasize, t_train

    ### Testing function
    def test_epoch(self, MRIdataloader):
        t_val=time.time()
        loss_fn=nn.MSELoss(reduction='sum')
        # Set evaluation mode
        self.model.eval()
        val_loss=0
        datasize=0

        with torch.no_grad(): # No need to track the gradients
            for image_batch, image_noisy in tqdm(MRIdataloader):
                result = self.model(image_noisy)
                val_loss += loss_fn(result, image_batch)
                datasize += image_batch.shape[0]
        val_loss=val_loss if not torch.is_tensor(val_loss) else val_loss.cpu().detach().numpy()
        t_val=time.time()-t_val
        return val_loss/datasize, t_val
    

    def train(self, dl_train, dl_val, max_epochs=100, keep_best_last=True):

        for current_epoch in range(self.last_epoch+1,max_epochs+1):            
            print('EPOCH %d/%d' % (current_epoch, max_epochs))
            ### Training (use the training function)
            train_loss, train_time = self.train_epoch(MRIdataloader=dl_train)
            ### Validation  (use the testing function)
            val_loss, val_time = self.test_epoch(MRIdataloader=dl_val)
            filename=f'{self.saved_models_path}/{self.model_name}_{self.radial_lines}lines_{self.rectype}_epoch{current_epoch}_{self.lr_type}LR_{self.lr_scheduler.get_last_lr()[0]}.pth'        
            minimum_value=self.df_history['val_loss'].min() if len(self.df_history)>0 else float('inf')

            if val_loss <= minimum_value:           
                self.save_model(filename, current_epoch, train_loss, val_loss)
                status='new best'
                self.best_model=filename
                print(f'new best loss: {val_loss}')
            else:                
                status='never best'

            self.df_history = pd.concat([self.df_history,
                                    pd.DataFrame({'epoch':[current_epoch],
                                                    'lr_type':[self.lr_type],
                                                    'learning_rate':[self.lr_scheduler.get_last_lr()[0]],#optimizer.param_groups[0]["lr"]],
                                                    'batch_size':[self.batch_size],
                                                    'train_loss':[train_loss],
                                                    'val_loss':[val_loss], 
                                                    'checkpoint':[filename], 
                                                    'status':[status],
                                                    'time_train':[train_time],
                                                    'time_val':[val_time]
                                                    })])

            self.df_history = self.df_history.loc[:, ~self.df_history.columns.str.contains('^Unnamed')]            
            
            self.df_history.to_csv(self.csv_filename)    
            if self.verbose:
                   print(self.df_history)
                   print(f'saving to {self.csv_filename}')
            if keep_best_last:
                self.erase_unwanted_models()
            self.last_model=filename       

        return self.df_history


    #https://towardsdatascience.com/a-batch-too-large-finding-the-batch-size-that-fits-on-gpus-aef70902a9f1
    def auto_find_batch_size(self, img_size=256, dataset_size: int=1000, max_batch_size: int = None,num_iterations: int = 5) -> int:
        torch.cuda.empty_cache()
        if self.verbose:
            print('Auto Batch Size...', end=' ')
        self.model.train(True)
        optimizer = torch.optim.Adam(self.model.parameters())
         
        batch_size = 1
        while True:
            if self.verbose:
                print(f'Trying batch = {batch_size}...', end=' || ')
            if max_batch_size is not None and batch_size >= max_batch_size:
                batch_size = max_batch_size
                break
            if batch_size >= dataset_size:
                batch_size = batch_size // 2
                break
            try:
                for _ in range(num_iterations):
                    # dummy inputs and targets
                    inputs = torch.rand((batch_size, self.number_of_channels(), img_size, img_size), device=self.device)
                    targets = torch.rand((batch_size, 1, img_size, img_size), device=self.device)
                    outputs = self.model(inputs)
                    loss = F.mse_loss(targets, outputs)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                batch_size *= 2
            except RuntimeError:
                batch_size //= 2
                break
        del self.model, optimizer
        torch.cuda.empty_cache()
        self.model=self.select_model() 
        self.load_model()
        self.batch_size=batch_size
        if self.verbose:
            print(f'Batch size = {batch_size}')
        return batch_size