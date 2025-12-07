from gpt_modeling import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import re
import os
import torch.distributed as dist
import datetime
import sys
import threading
from ctypes import *
import argparse
import json
import numpy as np

home_directory = os.path.expanduser( '~' )
sys.path.append(f"{home_directory}/FusionPipe/")
from src.scheduler_frontend import PyScheduler

def init_model(model_config):
    model=[]
    for i,j in model_config.items():
        if i=="em_tokn":
            mdl=nn.Embedding(j[0], j[1])
        elif i=="em_pos":
            mdl=nn.Embedding(j[0], j[1])
        elif i=="ln":
            mdl=nn.LayerNorm(j[0])
        elif i=="lm_head":
            mdl=nn.Linear(j[0],j[1])
        elif (re.search("decoder",i)).group()=="decoder": # 可能会报错
            mdl=Block(j[0],j[1])
        model.append(mdl)
    return model



class Stage:
    def __init__(self,ID,model,model_idx,learning_rate,device,batch_size):
        self.stage_ID=ID
        self.device=device
        self.model_idx=model_idx
        self.is_training=True
        self.sub_model= [model[i] for i in model_idx] # model.type:list
        self.optimizer_list= [optim.Adam(model[i].parameters(), lr=learning_rate) for i in model_idx]
        self.out_Y=[]
        self.out_X=[]
        self.out_x=torch.zeros(batch_size,128,128).to(device)
        self.grad_y=torch.zeros(batch_size,128,128).to(device)
        self.lossi=[]

    def to(self,device):
        for layer in self.sub_model:
            layer.to(device)
    
    def eval(self):
        for layer in self.sub_model:
            layer.eval()
    
    def train(self):
        for layer in self.sub_model:
            layer.train()

    def zero_grad(self):
        for optm in self.optimizer_list:
            optm.zero_grad()

    def update_out_Y(self,out_y):
        self.out_Y.append(out_y)

    def update_out_X(self,out_x):
        self.out_X.append(out_x)

    def forward(self,x):
        if self.stage_ID==1:
            B, T = x.shape
            # 定义词元的位置，形状为(T)
            pos = torch.arange(0, T, dtype=torch.long, device=x.device)
        # 词元语义特征
            tok_emb = self.sub_model[0](x)       # (B, T,  C)
        # 位置特征
            pos_emb = self.sub_model[1](pos)  # (   T,  C)
            x = tok_emb + pos_emb
            self.update_out_Y(x)
            for i in range(2,len(self.sub_model)):
                x=torch.tensor(data=x, dtype=torch.float, requires_grad=True, device=self.device)
                self.update_out_X(x)
                x=self.sub_model[i](x)
                self.update_out_Y(x)
            return x
        else:
            for layer in self.sub_model:
                x=torch.tensor(data=x, dtype=torch.float, requires_grad=True, device=self.device)
                self.update_out_X(x)
                x=layer(x) 
                self.update_out_Y(x)                
            return x
        
    def forward_send(self):

        dist.send(tensor=self.out_Y[-1],dst=self.stage_ID,tag=self.stage_ID)

    def forward_recv(self):
        dist.recv(tensor=self.out_x,src=self.stage_ID-2,tag=self.stage_ID-1)
        self.out_x.to(self.device)
        
    def backward_tail(self,labels):
        # print(self.out_Y[-1].shape)
        logits = self.out_Y[-1].transpose(-2, -1)
        loss = F.cross_entropy(logits, labels)
        self.lossi.append(loss.item())
        loss.backward()
        self.optimizer_list[-1].step()
        # print(self.out_X[-1].grad.shape)
        # print(type(self.out_X[0].grad))
        for i in range(2,len(self.out_Y)+1):
            # print(i)
            # print(len(self.out_X))
            # print(self.out_X[-i].shape)
            # # print(self.out_X[-i].grad.shape)
            # print(self.out_Y[-i].shape)
            self.out_Y[-i].backward(self.out_X[-i+1].grad) 
            self.optimizer_list[-i].step() 

    def backward(self):
        for i in range(1,len(self.out_Y)+1):
            if i==1:
                self.out_Y[-i].backward(self.grad_y)
                self.optimizer_list[-i].step() 
            else:               
                self.out_Y[-i].backward(self.out_X[-i+1].grad)
                self.optimizer_list[-i].step()
            
    def backward_send(self):
        dist.send(tensor=self.out_X[0].grad,dst=self.stage_ID-2,tag=self.stage_ID)

    def backward_recv(self):
        dist.recv(tensor=self.grad_y,src=self.stage_ID,tag=self.stage_ID+1)
        self.grad_y.to(self.device)

    
            

    def clear(self):
        self.out_Y.clear()
        self.out_X.clear()



def run_forward_head(stage,inputs,barriers):

    backend_lib = cdll.LoadLibrary(os.path.expanduser('~') + "/OpPipe/src/cuda_capture/libinttemp.so")

    stage.zero_grad()
    stage.forward(inputs)
    stage.forward_send()

    barriers[0].wait()

def run_forward_middle(stage,inputs,barriers):

    backend_lib = cdll.LoadLibrary(os.path.expanduser('~') + "/OpPipe/src/cuda_capture/libinttemp.so")

    stage.zero_grad()
    stage.forward_recv()
    stage.forward(inputs)
    stage.forward_send()

    barriers[0].wait()

def run_forward_middle(stage,inputs,barriers):

    backend_lib = cdll.LoadLibrary(os.path.expanduser('~') + "/OpPipe/src/cuda_capture/libinttemp.so")

    stage.zero_grad()
    stage.forward_recv()
    stage.forward(inputs)
    stage.forward_send()

    barriers[0].wait()    

def run_forward_tail(stage,inputs,barriers):

    backend_lib = cdll.LoadLibrary(os.path.expanduser('~') + "/OpPipe/src/cuda_capture/libinttemp.so")

    stage.zero_grad()
    stage.forward_recv()
    stage.forward(inputs)

    barriers[0].wait()    

# 通信域创建
env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "LOCAL_WORLD_SIZE")
    }
print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
dist.init_process_group(backend="NCCL", timeout=datetime.timedelta(seconds=30)) # gloo
global_rank = int(os.environ["RANK"])

if global_rank==0:
    DEVICE = 'cuda:0'
elif global_rank==1:
    DEVICE = 'cuda:1'
elif global_rank==2:
    DEVICE = 'cuda:2'
elif global_rank==3:
    DEVICE = 'cuda:3'

print(DEVICE)
MICRO_NUMBER=4

# 将数据分为训练集和测试集
tokenized = datasets.train_test_split(test_size=0.1, seed=1024, shuffle=True)
# 将文本转换为训练数据，里面包含inputs和labels
tokenized = tokenized.map(process, batched=True, remove_columns=datasets.column_names)
tokenized.set_format(type='torch', device=DEVICE)
print(tokenized['train']['inputs'].shape, tokenized['train']['labels'].shape)
# 构建数据读取器
train_loader = DataLoader(tokenized['train'], batch_size=batch_size, shuffle=True)
test_loader = DataLoader(tokenized['test'], batch_size=batch_size, shuffle=True)


model_config={"em_tokn":[98,128],"em_pos":[128,128],"decoder1":[128,8],"decoder2":[128,8],"decoder3":[128,8],
              "decoder4":[128,8],"decoder5":[128,8],"decoder6":[128,8],"decoder7":[128,8],"decoder8":[128,8],
              "decoder9":[128,8],"ln":[128],"lm_head":[128,98]}

gpt=init_model(model_config=model_config)
model_idx1=[0,1,2,3]
model_idx2=[4,5,6]
model_idx3=[7,8,9]
model_idx4=[10,11,12]
s1=Stage(1,gpt,model_idx1,learning_rate,DEVICE,batch_size)
s2=Stage(2,gpt,model_idx2,learning_rate,DEVICE,batch_size)
s3=Stage(3,gpt,model_idx3,learning_rate,DEVICE,batch_size)
s4=Stage(4,gpt,model_idx4,learning_rate,DEVICE,batch_size)

Stage_list=[s1,s2,s3,s4]

for i in range(len(Stage_list)):
    if i==global_rank:
        Stage_list[i].to(DEVICE)

# fuse 调度器
num_barriers = MICRO_NUMBER+1
barriers = [threading.Barrier(num_barriers) for i in range(MICRO_NUMBER)]
client_barrier = threading.Barrier(MICRO_NUMBER)
sched_lib = cdll.LoadLibrary(home_directory + "/orion/src/scheduler/scheduler_eval.so")
py_scheduler = PyScheduler(sched_lib, MICRO_NUMBER)

# 参数
parser = argparse.ArgumentParser()
parser.add_argument('--config_file', type=str, required=True,
                        help='path to the experiment configuration file')
args = parser.parse_args()
with open(args.config_file) as f:
        config_dict_list = json.load(f)

model_names = [config_dict['arch'] for config_dict in config_dict_list]
model_files = [config_dict['kernel_file'] for config_dict in config_dict_list]

additional_model_files = [config_dict['additional_kernel_file'] if 'additional_kernel_file' in config_dict else None for config_dict in config_dict_list]
num_kernels = [config_dict['num_kernels'] for config_dict in config_dict_list]
num_iters = [config_dict['num_iters'] for config_dict in config_dict_list]
train_list = [config_dict['args']['train'] for config_dict in config_dict_list]
additional_num_kernels = [config_dict['additional_num_kernels'] if 'additional_num_kernels' in config_dict else None  for config_dict in config_dict_list]
tids = []
threads = []


sched_thread = threading.Thread(
        target=py_scheduler.run_scheduler,
        args=(
            barriers,
            tids,
            model_names,
            model_files,
            additional_model_files,
            num_kernels,
            additional_num_kernels,
            num_iters,
            True,
            train_list
        )
    )
sched_thread.join()
print("sched joined!")

for i, data in tqdm(enumerate(train_loader, 0)):
    if global_rank==0:
        if Stage_list[0].is_training:
            inputs, labels = data['inputs'], data['labels']
            for _ in range(MICRO_NUMBER):
                thread = threading.Thread(target=run_forward_head, args=(Stage_list[0],inputs,barriers))
                thread.start()
                tids.append(thread.native_id)
                threads.append(thread)

            for thread in threads:
                thread.join()
            # Stage_list[0].backward_recv()
            # Stage_list[0].backward()

            Stage_list[0].clear()

        

    elif global_rank>0 and global_rank<len(Stage_list)-1:
        if Stage_list[global_rank].is_training:
            for _ in range(MICRO_NUMBER):
                thread = threading.Thread(target=run_forward_middle, args=(Stage_list[global_rank],Stage_list[global_rank].out_x,barriers))
                thread.start()
                tids.append(thread.native_id)
                threads.append(thread)

            for thread in threads:
                thread.join()

            # Stage_list[global_rank].backward_recv()
            # Stage_list[global_rank].backward()
            # Stage_list[global_rank].backward_send()

            Stage_list[global_rank].clear()

        
    else:
        if Stage_list[global_rank].is_training:
            inputs, labels = data['inputs'], data['labels']

            for _ in range(MICRO_NUMBER):
                thread = threading.Thread(target=run_forward_tail, args=(Stage_list[global_rank],Stage_list[global_rank].out_x,barriers))
                thread.start()
                tids.append(thread.native_id)
                threads.append(thread)

            for thread in threads:
                thread.join()
            # Stage_list[global_rank].backward_tail(labels)
            # Stage_list[global_rank].backward_send()

            Stage_list[global_rank].clear()
