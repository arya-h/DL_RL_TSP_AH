import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from time import time
from datetime import datetime
import numpy

from actor_embed import PtrNet1
from critic import PtrNet2
from env import Env_tsp
from config import Config, load_pkl, pkl_parser
from data import Generator

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True) # cached only for transductive learning
        self.conv2 = GCNConv(2 * out_channels, out_channels, cached=True) # cached only for transductive learning

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)
    
from torch_geometric.nn import GAE

# parameters
out_channels = 10
num_features = 2#dataset.num_features
epochs = 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model
#model_GAE = GAE(GCNEncoder(num_features, out_channels))
#model_GAE.load_state_dict(torch.load("./embedding_model/TRAIN_0412_16_58_step_GAE_GRAPH.pt", map_location = device))

# move to GPU (if available)

#model = model.to(device)

# torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.benchmark = True

def train_model(cfg, env, log_path = None):
	date = datetime.now().strftime('%m%d_%H_%M')
	if cfg.islogger:
		param_path = cfg.log_dir + '%s_%s_param.csv'%(date, cfg.task)# cfg.log_dir = ./Csv/
		print(f'generate {param_path}')
		with open(param_path, 'w') as f:
			f.write(''.join('%s,%s\n'%item for item in vars(cfg).items()))

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	act_model = PtrNet1(cfg)
	act_model.load_state_dict(torch.load("./model/train_random_nodes.pt", map_location = device))
    
    
    
	if cfg.optim == 'Adam':
		act_optim = optim.Adam(act_model.parameters(), lr = cfg.lr)
	if cfg.is_lr_decay:
		act_lr_scheduler = optim.lr_scheduler.StepLR(act_optim, 
						step_size=cfg.lr_decay_step, gamma=cfg.lr_decay)

	act_model = act_model.to(device)

	if cfg.mode == 'train':
		cri_model = PtrNet2(cfg)
		if cfg.optim == 'Adam':
			cri_optim = optim.Adam(cri_model.parameters(), lr = cfg.lr)
		if cfg.is_lr_decay:
			cri_lr_scheduler = optim.lr_scheduler.StepLR(cri_optim, 
						step_size = cfg.lr_decay_step, gamma = cfg.lr_decay)
		cri_model = cri_model.to(device)
		ave_cri_loss = 0.

	mse_loss = nn.MSELoss()
	dataset = Generator(cfg, env)
	dataloader = DataLoader(dataset, batch_size = cfg.batch, shuffle = True)

    


	ave_act_loss, ave_L = 0., 0.
	min_L, cnt = 1e7, 0
	t1 = time()
	# for i, inputs in tqdm(enumerate(dataloader)):
	#for i, inputs in enumerate(dataloader):
	for i, inputs in enumerate(dataloader):
		inputs = inputs.to(device)
        
        
		#train_pos_edge_index = data.train_pos_edge_index.to(device)
		#embed = model_GAE.encode(inputs)
		pred_tour, ll = act_model(inputs, device)
		#pred_tour, ll = act_model(embed, device)
        
        #single tour is in the following format
        #tensor([3, 1, 4, 8, 6, 7, 2, 0, 9, 5])
        
        #tour is in the following format
        #tensor(-3.2278, grad_fn=<SelectBackward0>)
        
        #coordinates are in inputs variable
        
        
        #for gr in pred_tour:
            #data.x = features
        

		#print(pred_tour[0])
		#print(ll[0])
        
        
		real_l = env.stack_l_fast(inputs, pred_tour)
		if cfg.mode == 'train':
			pred_l = cri_model(inputs, device)
			cri_loss = mse_loss(pred_l, real_l.detach())
			cri_optim.zero_grad()
			cri_loss.backward()
			nn.utils.clip_grad_norm_(cri_model.parameters(), max_norm = 1., norm_type = 2)
			cri_optim.step()
			if cfg.is_lr_decay:
				cri_lr_scheduler.step()

		adv = real_l.detach() - pred_l.detach()
		act_loss = (adv * ll).mean()
		act_optim.zero_grad()
		act_loss.backward()
		nn.utils.clip_grad_norm_(act_model.parameters(), max_norm = 1., norm_type = 2)
		act_optim.step()
		if cfg.is_lr_decay:
			act_lr_scheduler.step()

		ave_act_loss += act_loss.item()
		if cfg.mode == 'train':
			ave_cri_loss += cri_loss.item()
		ave_L += real_l.mean().item()

		if i % 5 == 0:
            #find smallest tour
            
			env.show(env.get_nodes(), pred_tour[0])
			#print(pred_tour[''])

		if i % cfg.log_step == 0:
			t2 = time()
			if cfg.mode == 'train':	
				print('step:%d/%d, actic loss:%1.3f, critic loss:%1.3f, L:%1.3f, %dmin%dsec'%(i, cfg.steps, ave_act_loss/(i+1), ave_cri_loss/(i+1), ave_L/(i+1), (t2-t1)//60, (t2-t1)%60))
				if cfg.islogger:
					if log_path is None:
						log_path = cfg.log_dir + '%s_%s_train.csv'%(date, cfg.task)#cfg.log_dir = ./Csv/
						with open(log_path, 'w') as f:
							f.write('step,actic loss,critic loss,average distance,time\n')
					else:
						with open(log_path, 'a') as f:
							f.write('%d,%1.4f,%1.4f,%1.4f,%dmin%dsec\n'%(i, ave_act_loss/(i+1), ave_cri_loss/(i+1), ave_L/(i+1), (t2-t1)//60, (t2-t1)%60))
			
			
			if(ave_L/(i+1) < min_L):
				min_L = ave_L/(i+1)
				
			# else:
			# 	cnt += 1
			# 	print(f'cnt: {cnt}/20')
			# 	if(cnt >= 20):
			# 		print('early stop, average cost cant decrease anymore')
			# 		if log_path is not None:
			# 			with open(log_path, 'a') as f:
			# 				f.write('\nearly stop')
			# 		break
			t1 = time()
	if cfg.issaver:		
		torch.save(act_model.state_dict(), cfg.model_dir + '%s_%s_step%d_act.pt'%(cfg.task, date, i))#'cfg.model_dir = ./Pt/'
		print('save model...')
