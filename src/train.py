import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
import pickle
from tokenizer import Tokenizer
import time

from utils.config import Config
from utils.utils_func import *
from utils.utils_data import DLoader
from models.model import ConvBERT



class Trainer:
    def __init__(self, config:Config, device:torch.device, mode:str, continuous:int):
        self.config = config
        self.device = device
        self.mode = mode
        self.continuous = continuous
        self.dataloaders = {}

        # if continuous, load previous training info
        if self.continuous:
            with open(self.config.loss_data_path, 'rb') as f:
                self.loss_data = pickle.load(f)

        # path, data params
        self.base_path = self.config.base_path
        self.model_path = self.config.model_path
        self.data_path = self.config.dataset_path
 
        # train params
        self.batch_size = self.config.batch_size
        self.epochs = self.config.epochs
        self.lr = self.config.lr

        # define tokenizer
        self.tokenizer = Tokenizer(self.config)

        # dataloader
        torch.manual_seed(999)  # for reproducibility
        if self.mode == 'train':
            self.dataset = {s: DLoader(load_dataset(p), self.tokenizer, self.config) for s, p in self.data_path.items()}
            self.dataloaders = {
                s: DataLoader(d, self.batch_size, shuffle=True) if s == 'train' else DataLoader(d, self.batch_size, shuffle=False)
                for s, d in self.dataset.items()}
        else:
            self.dataset = {s: DLoader(load_dataset(p), self.tokenizer, self.config) for s, p in self.data_path.items() if s == 'test'}
            self.dataloaders = {s: DataLoader(d, self.batch_size, shuffle=False) for s, d in self.dataset.items() if s == 'test'}

        # model, optimizer, loss
        self.model = ConvBERT(self.config, self.tokenizer, self.device).to(self.device)
        self.nsp_criterion = nn.CrossEntropyLoss()
        self.mlm_criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
    
        if self.mode == 'train':
            total_steps = len(self.dataloaders['train']) * self.epochs
            pct_start = 20000 / total_steps
            final_div_factor = self.lr / 25 / 1e-6
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999))
            self.scheduler = OneCycleLR(self.optimizer, max_lr=self.lr, total_steps=total_steps, pct_start=pct_start, final_div_factor=final_div_factor)
            if self.continuous:
                self.check_point = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(self.check_point['model'])
                self.optimizer.load_state_dict(self.check_point['optimizer'])
                del self.check_point
                torch.cuda.empty_cache()
        else:
            self.check_point = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(self.check_point['model'])    
            self.model.eval()
            del self.check_point
            torch.cuda.empty_cache()

        
    def training(self):
        early_stop = 0
        best_val_loss = float('inf') if not self.continuous else self.loss_data['best_val_loss']
        best_epoch_info = 0 if not self.continuous else self.loss_data['best_epoch']
        self.loss_data = {
            'train_history': {'loss': {'total': [], 'nsp': [], 'mlm': []}, 'nsp_acc': []}, \
            'val_history': {'loss': {'total': [], 'nsp': [], 'mlm': []}, 'nsp_acc': []}
            }

        for epoch in range(self.epochs):
            start = time.time()
            print(epoch+1, '/', self.epochs)
            print('-'*10)
            for phase in ['train', 'val']:
                print('Phase: {}'.format(phase))
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                total_loss, total_nsp_loss, total_mlm_loss, total_nsp_acc = 0, 0, 0, 0
                for i, (x, segment, nsp_label, mlm_label) in enumerate(self.dataloaders[phase]):
                    batch_size = x.size(0)
                    x, segment, nsp_label, mlm_label = x.to(self.device), segment.to(self.device), nsp_label.to(self.device), mlm_label.to(self.device)
                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase=='train'):
                        _, (nsp_output, mlm_output) = self.model(x, segment)
                        nsp_loss = self.nsp_criterion(nsp_output, nsp_label)
                        mlm_loss = self.mlm_criterion(mlm_output.reshape(-1, mlm_output.size(-1)), mlm_label.reshape(-1))
                        loss = nsp_loss + mlm_loss

                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()
                            self.scheduler.step()

                        nsp_acc = torch.sum(torch.argmax(nsp_output, dim=-1).detach().cpu() == nsp_label.detach().cpu()) / batch_size

                    total_loss += loss.item() * batch_size
                    total_nsp_loss += nsp_loss.item() * batch_size
                    total_mlm_loss += mlm_loss.item() * batch_size
                    total_nsp_acc += nsp_acc * batch_size

                    if i % 1000 == 0:
                        print('Epoch {}: {}/{} step loss: {} (nsp: {}, mlm: {}), nsp acc: {}'.format(epoch+1, i, len(self.dataloaders[phase]), loss.item(), nsp_loss.item(), mlm_loss.item(), nsp_acc))
                
                dataset_len = len(self.dataloaders[phase].dataset)
                epoch_loss = total_loss / dataset_len
                epoch_nsp_loss = total_nsp_loss / dataset_len
                epoch_mlm_loss = total_mlm_loss / dataset_len
                epoch_nsp_acc = total_nsp_acc / dataset_len
                print('{} loss: {:4f} (nsp: {}, mlm: {}), nsp acc: {}\n'.format(phase, epoch_loss, epoch_nsp_loss, epoch_mlm_loss, epoch_nsp_acc))

                if phase == 'train':
                    self.loss_data['train_history']['loss']['total'].append(epoch_loss)
                    self.loss_data['train_history']['loss']['nsp'].append(epoch_nsp_loss)
                    self.loss_data['train_history']['loss']['mlm'].append(epoch_mlm_loss)
                    self.loss_data['train_history']['nsp_acc'].append(epoch_nsp_acc)

                elif phase == 'val':
                    self.loss_data['val_history']['loss']['total'].append(epoch_loss)
                    self.loss_data['val_history']['loss']['nsp'].append(epoch_nsp_loss)
                    self.loss_data['val_history']['loss']['mlm'].append(epoch_mlm_loss)
                    self.loss_data['val_history']['nsp_acc'].append(epoch_nsp_acc)
            
                    # save best model
                    early_stop += 1
                    if  epoch_loss < best_val_loss:
                        early_stop = 0
                        best_val_loss = epoch_loss
                        best_epoch = best_epoch_info + epoch + 1
                        self.loss_data['best_epoch'] = best_epoch
                        self.loss_data['best_val_loss'] = best_val_loss
                        save_checkpoint(self.model_path, self.model, self.optimizer)

            print("time: {} s\n".format(time.time() - start))
            print('\n'*2)

            # early stopping
            if early_stop == self.config.early_stop_criterion:
                break

        print('best val loss: {:4f}, best epoch: {:d}\n'.format(best_val_loss, best_epoch))

        return self.loss_data


    def test(self, phase):
        with torch.no_grad():
            self.model.eval()
            total_loss, total_nsp_loss, total_mlm_loss, total_nsp_acc = 0, 0, 0, 0
            for x, segment, nsp_label, mlm_label in self.dataloaders[phase]:
                batch_size = x.size(0)
                x, segment, nsp_label, mlm_label = x.to(self.device), segment.to(self.device), nsp_label.to(self.device), mlm_label.to(self.device)
                _, (nsp_output, mlm_output) = self.model(x, segment)
                nsp_loss = self.nsp_criterion(nsp_output, nsp_label)
                mlm_loss = self.mlm_criterion(mlm_output.reshape(-1, mlm_output.size(-1)), mlm_label.reshape(-1))
                loss = nsp_loss + mlm_loss
                nsp_acc = torch.sum(torch.argmax(nsp_output, dim=-1).detach().cpu() == nsp_label.detach().cpu()) / batch_size

                total_loss += loss.item() * batch_size
                total_nsp_loss += nsp_loss.item() * batch_size
                total_mlm_loss += mlm_loss.item() * batch_size
                total_nsp_acc += nsp_acc * batch_size

        dataset_len = len(self.dataloaders[phase].dataset)
        epoch_loss = total_loss / dataset_len
        epoch_nsp_loss = total_nsp_loss / dataset_len
        epoch_mlm_loss = total_mlm_loss / dataset_len
        epoch_nsp_acc = total_nsp_acc / dataset_len
        print('{} loss: {:4f} (nsp: {}, mlm: {}), nsp acc: {}\n'.format(phase, epoch_loss, epoch_nsp_loss, epoch_mlm_loss, epoch_nsp_acc))