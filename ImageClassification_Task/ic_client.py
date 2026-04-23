import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import multiprocessing
from threading import Thread
from typing import Optional
from utils.connections import is_socket_closed
from utils.connections import send_object
from utils.connections import get_object
import pickle
import queue
import struct
import numpy as np
from tqdm.auto import tqdm

# loss and metrics
from sklearn.metrics import balanced_accuracy_score
from torchmetrics.functional.classification import f1_score
from ImageClassification_Task.focal_loss_fn import FocalLoss


class Client(Thread):
    def __init__(self, id, *args, **kwargs):
        super(Client, self).__init__(*args, **kwargs)
        self.id = id
        self.flag=0
        self.test_flag=0
        self.front_model = []
        
        self.back_model = []
        self.center_model = []
        self.losses = []


        self.current_keys=[]
        self.target_mappings={}
        self.activation_mappings={}
        self.data_key=0
        self.kv_flag=0
        self.kv_test_flag=0

        self.test_target_mappings={}
        self.test_activation_mappings={}
        self.test_data_key=0

        self.train_dataset = None
        self.test_dataset = None
        self.main_test_dataset = None
        self.train_DataLoader = None
        self.test_DataLoader = None
        self.main_test_DataLoader = None
        self.socket = None
        self.server_socket = None
        self.train_batch_size = None
        self.test_batch_size = None
        self.iterator = None
        self.test_iterator=None
        self.activations1 = None
        self.remote_activations1 = None
        self.outputs = None
        self.loss = None

        self.train_loss = 0
        self.test_loss = 0

        self.criterion = None
        self.data = None
        self.targets = None
        self.batchkeys = None
        self.test_batchkeys = None
        self.key = None
        self.test_key = None
        self.n_correct = 0
        self.n_samples = 0
        self.front_optimizer = None
        self.back_optimizer = None
        self.back_scheduler = None
        self.train_f1 = []
        self.test_f1 = []

        self.train_preds = []
        self.train_targets = []
        self.test_preds = []
        self.test_targets = []

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.loss_fn = nn.CrossEntropyLoss()
        #self.loss_fn = FocalLoss()
        self.use_fp16 = False

        # --- Mixup (Option B) state ---
        # Populated during KV store creation:
        #   key   = id_a  (string, e.g. "0-0-37")
        #   value = (target_b, lam)  – secondary label and mixing weight
        self.mixup_map: dict = {}
        # Set by the trainer just before calculate_loss() is called each iteration.
        # Both are per-sample tensors (length = batch size) so each sample uses
        # its own λ from when it was mixed during KV-store population.
        self.mixup_lam = None          # 1-D float tensor [B] of per-sample λ values
        self.mixup_targets_b = None    # 1-D long  tensor [B] of secondary labels

    @torch.no_grad()
    def balanced_accuracy(self,preds,targets):
        targets = targets.reshape(-1).numpy()
        preds = np.argmax(preds.numpy(), axis=1)
        return balanced_accuracy_score(targets, preds)
     
    @torch.no_grad()    
    def normal_accuracy(self, preds, targets):
	    """
	    Calculates normal accuracy for classification tasks
	    """
	    targets = targets.reshape(-1).numpy()
	    preds = np.argmax(preds.numpy(), axis=1)
	    correct = np.sum(preds == targets)
	    total = len(targets)
	    accuracy = correct / total
	    return accuracy
	    
    @torch.no_grad()
    def run_metric(self,preds,targets):
        #print("Run metric")
        #print("Shapes - preds:", preds.shape, "targets:", targets.shape)
        #print(preds.shape[1])
        #print(torch.argmax(preds,dim=1).float())
        #print(targets.squeeze())
        return f1_score(
            #preds=torch.argmax(preds,dim=1).float(),
            preds=preds.argmax(dim=1),
            target=targets,
            task='multiclass',
            num_classes=10
        )
    
    @torch.no_grad()
    def get_main_metric(self,mode='train'):
        """
        calculates main metric to use and then resets over epoch
        """
        if mode=='train':
            preds= torch.cat(self.train_preds,dim=0)
            targets = torch.cat(self.train_targets, dim=0)
            #preds = torch.vstack(self.train_preds)
            #targets = torch.vstack(self.train_targets)
            #print("Shapes - preds:", preds.shape, "targets:", targets.shape)
            bal_acc = self.normal_accuracy(preds, targets)
            #print(bal_acc)
            f1_macro = f1_score(
                preds=torch.argmax(preds,dim=1).float(),
                target=targets.squeeze(),
                task='multiclass',
                num_classes=10,
                average='macro'
            ).item()
            self.train_targets = []
            self.train_preds = []
        elif mode=='test':
            preds= torch.cat(self.test_preds,dim=0)
            targets = torch.cat(self.test_targets, dim=0)
            #preds = torch.vstack(self.test_preds)
            #targets = torch.vstack(self.test_targets)
            #print("Shapes - preds:", preds.shape, "targets:", targets.shape)
            bal_acc = self.normal_accuracy(preds, targets)
            #print(bal_acc)
            f1_macro = f1_score(
                preds=torch.argmax(preds,dim=1).float(),
                target=targets.squeeze(),
                task='multiclass',
                num_classes=10,
                average='macro'
            ).item()
            self.test_preds = []
            self.test_targets = []

        return bal_acc, f1_macro


    def backward_back(self):
        self.loss.backward()
           
    def backward_front(self):
        self.activations1.backward(self.remote_activations1.grad)

    
    def calculate_loss(self, mode='train'):
        """
        Calculates the loss for the current batch.
        If mixup is active, computes a PER-SAMPLE weighted CE loss so that
        each sample uses exactly the λ that was used to blend its activation
        during KV-store population:
            loss_i = λ_i · CE(pred_i, target_a_i) + (1-λ_i) · CE(pred_i, target_b_i)
            loss   = mean(loss_i)          ← averaged over the batch
        Otherwise falls back to standard single-label CE.
        """
        if self.mixup_lam is not None and self.mixup_targets_b is not None:
            # self.mixup_lam      : float tensor [B] — per-sample λ
            # self.mixup_targets_b: long  tensor [B] — per-sample secondary label
            lam = self.mixup_lam.to(self.outputs.device)          # [B]
            # CE loss per sample (reduction='none' → shape [B])
            loss_fn_none = nn.CrossEntropyLoss(reduction='none')
            loss_a = loss_fn_none(self.outputs, self.targets.long())           # [B]
            loss_b = loss_fn_none(self.outputs, self.mixup_targets_b.long())  # [B]
            self.loss = (lam * loss_a + (1.0 - lam) * loss_b).mean()
        else:
            self.loss = self.loss_fn(self.outputs, self.targets.long())

        if mode == 'train':
            self.train_loss += self.loss.item()
        elif mode == 'test':
            self.test_loss += self.loss.item()

    @torch.no_grad()
    def calculate_train_metric(self):
        preds = self.outputs
        targets = self.targets
        #print("Calculate train metric")
        #print("Shapes - preds:", preds.shape, "targets:", targets.shape)
        self.train_preds.append(preds.cpu())
        self.train_targets.append(targets.cpu())
        f1 = self.run_metric(preds.cpu(),targets.cpu())
        return f1
    
    @torch.no_grad()
    def calculate_test_metric(self):
        preds = self.outputs
        targets = self.targets
        self.test_preds.append(preds.cpu())
        self.test_targets.append(targets.cpu())
        f1 = self.run_metric(preds.cpu(),targets.cpu())
        return f1
    
    
    

    def connect_server(self, host='localhost', port=8000, BUFFER_SIZE=4096):
        self.socket, self.server_socket = multiprocessing.Pipe()
        print(f"[*] Client {self.id} connecting to {host}")


    def create_DataLoader(self, train_batch_size, test_batch_size):
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        
        # TORCH DATALOADERS
        self.train_DataLoader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        self.test_DataLoader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        self.main_test_DataLoader = torch.utils.data.DataLoader(
            self.main_test_dataset,
            batch_size=self.test_batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )

    def disconnect_server(self) -> bool:
        if not is_socket_closed(self.socket):
            self.socket.close()
            return True
        else:
            return False


    def forward_back(self):
        # self.back_model.to(self.device)
        self.outputs = self.back_model(self.remote_activations2)
        #print("Size of clinet_activations1:", self.remote_activations2.size())
        #print("Size of outputs",self.outputs.size())
        
    def forward_back_personalise(self):
        batch_data = next(self.iterator)
        self.data, self.targets, self.key = batch_data['image'].to(self.device), batch_data['label'].to(self.device), batch_data['id']
        valid_keys = [key for key in self.key if key in self.activation_mappings]
        activations_list = [self.activation_mappings[key] for key in valid_keys]
        activations_array = np.array(activations_list)
        x2 = torch.tensor(activations_array, device=self.device)
        #print("Size of x2",x2.size())
        self.outputs = self.back_model(x2)
        #print("Size of output",self.outputs.size())
        
    def forward_back_personalise_test(self):
        batch_data = next(self.test_iterator)
        self.data, self.targets, self.key = batch_data['image'].to(self.device), batch_data['label'].to(self.device), batch_data['id']
        valid_keys = [key for key in self.key if key in self.activation_mappings]
        activations_list = [self.activation_mappings[key] for key in valid_keys]
        activations_array = np.array(activations_list)
        x2 = torch.tensor(activations_array, device=self.device)
        #print("Size of x2",x2.size())
        self.outputs = self.back_model(x2)
        #print("Size of output",self.outputs.size())
        
        


    def forward_front(self):
        batch_data = next(self.iterator)
        self.data, self.targets, self.key = batch_data['image'].to(self.device), batch_data['label'].to(self.device), batch_data['id'].to(self.device)
        #print(self.key)
        
        # self.front_model.to(self.device)
        self.activations1 = self.front_model(self.data)
        self.remote_activations1 = self.activations1.detach().requires_grad_(True)

        # return self.activations1
                    
    def forward_front_key_value(self):
        batch_data = next(self.iterator)
        #self.data, self.targets = batch_data['image'].to(self.device), batch_data['label'].to(self.device)
        self.data, self.targets, self.key = batch_data['image'].to(self.device), batch_data['label'].to(self.device), batch_data['id']
        #print("Label", self.targets)
        #print("keys",self.key)
        # self.front_model.to(self.device)
        if self.kv_flag==1:
            self.activations1 = self.front_model(self.data)
            #print("Size of data:", self.data.size())
            #print("Size of clinet_activations1:", self.activations1.size())
            self.remote_activations1 = self.activations1.detach().requires_grad_(True)


    def forward_front_key_value_test(self):
        #print("forward method")
        #print("self.kv_flag",self.kv_test_flag)
        batch_data = next(self.test_iterator)
        self.data, self.targets , self.test_key= batch_data['image'].to(self.device), batch_data['label'].to(self.device), batch_data['id']
        #print("target", self.targets)
        #print("keys",self.test_key)
        # self.front_model.to(self.device)
        #self.activations1 = self.front_model(self.data)
        if self.kv_test_flag==1:
            self.activations1 = self.front_model(self.data)
            #print("Size of data:", self.data.size())
            #print("Size of clinet_activations1:", self.activations1.size())
            self.remote_activations1 = self.activations1.detach().requires_grad_(True)


    def get_model(self):
        model = get_object(self.socket)
        self.front_model = model['front']
        self.back_model = model['back']


    def get_remote_activations1_grads(self):
        self.remote_activations1.grad = get_object(self.socket)


    def get_remote_activations2(self):
        # Receive activation from server; cast only when FP16 transport is enabled.
        remote_activations2 = get_object(self.socket)
        if self.use_fp16:
            # Server sends FP16 only when transport simulation is enabled.
            remote_activations2 = remote_activations2.float()
        self.remote_activations2 = remote_activations2.requires_grad_(True)


    def idle(self):
        pass


    def send_remote_activations1(self):
        send_object(self.socket, self.remote_activations1)
    

    def send_remote_activations2_grads(self):
        # Send gradient to server; cast only when FP16 transport is enabled.
        remote_grad = self.remote_activations2.grad
        if self.use_fp16:
            # Client sends FP16 gradients only when transport simulation is enabled.
            remote_grad = remote_grad.half()
        send_object(self.socket, remote_grad)


    def step_front(self):
        self.front_optimizer.step()
        

    def step_back(self):
        self.back_optimizer.step()


    def zero_grad_front(self):
        self.front_optimizer.zero_grad()
        

    def zero_grad_back(self):
        self.back_optimizer.zero_grad()

