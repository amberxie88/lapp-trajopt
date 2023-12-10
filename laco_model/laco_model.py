import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import inspect
import re
import utils
from clip_utils import load_model, load_tokenizer, load_vision_model
from transformer_utils import build_2d_sincos_posemb, pair, CrossBlock, Block, LayerNorm
from typing import Union, Tuple, Optional
from einops import rearrange, repeat

from cd_model.base_model import BaseModel

# adapted from Multi-MAE https://github.com/EPFL-VILAB/MultiMAE/blob/11167059599e563a0edb9cb48e1dc8ab45ad4b92/multimae/input_adapters.py
class PatchedInputAdapter(nn.Module):
    """Adapter for spatial inputs, like images or feature maps.
    Creates tokens from patches over the image.
    :param num_channels: Number of input channels of the image/feature map
    :param patch_size_full: Int or tuple of the patch size over the full image size.
        Patch size for smaller inputs will be computed accordingly.
    :param dim_tokens: Dimension of output tokens. Can be set using init method.
    :param sincos_pos_emb: Set to True (default) to use fixed 2D sin-cos positional embeddings
    :param learnable_pos_emb: Set to True to learn positional embeddings instead
    :param image_size: Default image size. Used to initialize size of positional embeddings.
    """
    def __init__(self,
                 num_channels: int,
                 patch_size_full: Union[int, Tuple[int,int]],
                 dim_tokens: Optional[int] = None,
                 sincos_pos_emb: bool = True,
                 learnable_pos_emb: bool = False,
                 image_size: Union[int, Tuple[int]] = 256):

        super().__init__()
        self.num_channels = num_channels
        self.patch_size_full = pair(patch_size_full)
        self.dim_tokens = dim_tokens
        self.sincos_pos_emb = sincos_pos_emb
        self.learnable_pos_emb = learnable_pos_emb
        self.image_size = pair(image_size)
        self.num_patches = (self.image_size[0] // patch_size_full) * (self.image_size[1] // patch_size_full)

        self.P_H = max(1, self.patch_size_full[0])
        self.P_W = max(1, self.patch_size_full[1])

        if self.dim_tokens is not None:
            self.init(dim_tokens=dim_tokens)

    def init(self, dim_tokens: int = 768):
        """
        Initialize parts of encoder that are dependent on dimension of tokens.
        Should be called when setting up MultiMAE.
        :param dim_tokens: Dimension of tokens
        """
        self.dim_tokens = dim_tokens

        # Task embedding identifying from which task a given token comes from
        # Fixed-size positional embeddings. Can be interpolated to different input sizes
        h_posemb = self.image_size[0] // self.P_H
        w_posemb = self.image_size[1] // self.P_W
        if self.sincos_pos_emb:
            self.pos_emb = build_2d_sincos_posemb(h=h_posemb, w=w_posemb, embed_dim=self.dim_tokens)
            self.pos_emb = nn.Parameter(self.pos_emb, requires_grad=self.learnable_pos_emb)
        else:
            self.pos_emb = nn.Parameter(torch.zeros(1, self.dim_tokens, h_posemb, w_posemb))
            trunc_normal_(self.pos_emb, std=0.02)

        # Image -> tokens projection
        self.proj = nn.Conv2d(
            in_channels=self.num_channels, out_channels=self.dim_tokens,
            kernel_size=(self.P_H, self.P_W), stride=(self.P_H, self.P_W)
        )

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_emb'}

    def forward(self, x):
        """
        Forward pass through input adapter, transforming image to sequence of tokens.
        Adds task and positional encodings.
        :param x: Input image tensor
        """
        B, C, H, W = x.shape
        assert self.dim_tokens is not None, 'Need to call init(dim_tokens) function first'
        assert (H % self.P_H == 0) and (W % self.P_W == 0), f'Image sizes {H}x{W} must be divisible by patch sizes {self.P_H}x{self.P_W}'
        N_H, N_W = H // self.P_H, W // self.P_W # Number of patches in height and width

        # Create patches [B, C, H, W] -> [B, (P_H*P_W), C]
        x_patch = rearrange(self.proj(x), 'b d nh nw -> b (nh nw) d')

        # Create positional embedding
        x_pos_emb = F.interpolate(self.pos_emb, size=(N_H, N_W), mode='bicubic', align_corners=False)
        x_pos_emb = rearrange(x_pos_emb, 'b d nh nw -> b (nh nw) d')

        # Add patches and positional embeddings
        x = x_patch + x_pos_emb

        return x

class LangAttentionModel(BaseModel):
	"""
	Transformer-based Collision Detection module that takes in language, image, and state.
	"""

	def __init__(self, name, device, lr, state_hidden_dim, obs_n_filters, net_hidden_dim,
				 shared_hidden_dim, obs_encoder_id, loss, threshold, state_setting, 
				 encoder, model_max_length, freeze_language_encoder,
				 modality_embedding, lang_embedding_dim, patch_size, token_dim, use_mv,
				 dropout, n_layer, cross_attention, bias, n_head, weight_decay, pred_coll_loss, alpha_coll_loss,
				 pred_net_id, attention_feat, n_obs_attn_layer, use_transformer_pos_emb,
				 scheduler, schedule_every_step, state_dim, pred_obj_in_scene_loss, alpha_pred_obj_in_scene_loss,
				 normalize_state):
		super(LangAttentionModel, self).__init__()
		self.device = device
		self.obs_encoder_id = obs_encoder_id

		self.threshold = threshold # past threshold indicates collision

		self.modality_embedding = modality_embedding
		if modality_embedding == True:
			self.obs_type_embedding = nn.Parameter(init.normal_(torch.empty((1, 1, token_dim)), std=0.02))
			self.lang_type_embedding = nn.Parameter(init.normal_(torch.empty((1, 1, token_dim)), std=0.02))
			self.state_type_embedding = nn.Parameter(init.normal_(torch.empty((1, 1, token_dim)), std=0.02))
			
		assert state_setting in ['current', 'init_current', 'delta']
		self.state_setting = state_setting
		self.state_dim = state_dim
		if isinstance(state_hidden_dim, int):
			raise NotImplementedError
		else:
			model = [nn.Linear(state_dim, state_hidden_dim[0]), nn.ReLU()]
			for i in range(0, len(state_hidden_dim) - 1):
				model += [nn.Linear(state_hidden_dim[i], state_hidden_dim[i+1]), nn.ReLU()] 
			self.state_encoder = nn.Sequential(*model).to(self.device) 
			self.state_linear = nn.Linear(state_hidden_dim[-1], token_dim).to(self.device)

		if self.obs_encoder_id == "mae":
			pass
		elif self.obs_encoder_id == "multimae_patch":
			self.obs_tokenizer = PatchedInputAdapter(3, patch_size, token_dim, sincos_pos_emb=True, learnable_pos_emb=False, image_size=256).to(self.device)
		elif self.obs_encoder_id in ["clip16", "clip14", "clip16_scratch"]:
			# could enforce 16 or 14 alignment more rigorously if desired
			encoder = self.obs_encoder_id
			# assert "clip16" in encoder or "clip14" in encoder
			self.vision_model = load_vision_model(self.obs_encoder_id).to(self.device)
		else:
			raise NotImplementedError

		# transformer: n_layer = num_self_attention_layers + 1 (cross_attention_layer)
		self.transformer = nn.ModuleDict(dict(
            drop = nn.Dropout(dropout),
            h = nn.ModuleList([Block(token_dim, n_head, bias, dropout) for _ in range(n_layer-int(cross_attention))]),
            cross = CrossBlock(token_dim, n_head, bias, dropout) if cross_attention else None,
            ln_f = LayerNorm(token_dim, bias=bias),
        )).to(self.device)
		self.cross_attention = cross_attention
		self.attention_feat = attention_feat
		self.token_dim = token_dim
		if self.token_dim != 768:
			self.obs_linear = nn.Linear(768, token_dim)

		# additional observation attention layers
		self.use_transformer_pos_emb = use_transformer_pos_emb
		if self.use_transformer_pos_emb:
		    pos_emb = build_2d_sincos_posemb(h=14, w=14, embed_dim=token_dim)
		    pos_emb = nn.Parameter(pos_emb, requires_grad=False)
		    pos_emb = F.interpolate(pos_emb, size=(14, 14), mode='bicubic', align_corners=False).to(self.device)
		    self.transformer_pos_emb = rearrange(pos_emb, 'b d nh nw -> b (nh nw) d')
		self.use_mv = use_mv 
		assert not self.use_mv or self.use_mv != (n_obs_attn_layer > 0), "use_mv and n_obs_attn_layer must be mutually exclusive"
		self.n_obs_attn_layer = n_obs_attn_layer
		if self.n_obs_attn_layer > 0:
			self.obs_attn = nn.ModuleList([Block(token_dim, n_head, bias, dropout) for _ in range(self.n_obs_attn_layer)]).to(self.device)

		# auxiliary losses
		self.pred_coll_loss, self.alpha_coll_loss = pred_coll_loss, alpha_coll_loss
		self.pred_obj_in_scene_loss, self.alpha_pred_obj_in_scene_loss = pred_obj_in_scene_loss, alpha_pred_obj_in_scene_loss
		
		# prediction head
		if pred_net_id == "linear_to_sigmoid":
			self.pred_head = nn.Sequential(nn.Linear(token_dim, 1+int(pred_coll_loss)+int(pred_obj_in_scene_loss), bias=False), nn.Sigmoid()).to(self.device)
		elif pred_net_id == "layers":
			model = [nn.Linear(token_dim, net_hidden_dim[0]), nn.ReLU()]
			for i in range(0, len(net_hidden_dim) - 1):
				model += [nn.Linear(net_hidden_dim[i], net_hidden_dim[i+1]), nn.ReLU()]
			model += [nn.Linear(net_hidden_dim[-1], 1+int(pred_coll_loss)+int(pred_obj_in_scene_loss), bias=False), nn.Sigmoid()] 
			self.pred_head = nn.Sequential(*model).to(self.device) 
		else:
			raise NotImplementedError

		# language model 
		self.tokenizer = load_tokenizer(encoder)
		self.tokenizer.model_max_length = model_max_length
		model = load_model(encoder)
		self.model = model.to(self.device)
		if "clip16" in encoder or encoder == "clip":
			self.proj_instr_encoder = nn.Linear(512, token_dim).to(self.device)
		elif "clip14" in encoder:
			self.proj_instr_encoder = nn.Linear(768, token_dim).to(self.device)
		else:
			raise NotImplementedError

		# networks to save and optimize
		self.networks = ['state_encoder', 'state_linear', 
						'pred_head',
						'proj_instr_encoder', 'transformer']
		if self.obs_encoder_id == "multimae_patch":
			self.networks.append('obs_tokenizer')
		if self.n_obs_attn_layer > 0:
			self.networks.append('obs_attn')
		if self.modality_embedding:
			self.networks.extend(['obs_type_embedding', 'lang_type_embedding', 'state_type_embedding'])
		if self.token_dim != 768:
			self.networks.append('obs_linear')
		if self.obs_encoder_id in ["clip16_scratch"]:
			self.networks.append('vision_model')
		self.optims = [self.configure_optims(lr, weight_decay)]
		self.schedulers = self.configure_schedulers(scheduler, schedule_every_step)
		self.normalize_state = normalize_state
		if self.normalize_state:
			self.state_mean = torch.tensor([-3.44671526, -0.01341378, -2.8631923 ,  1.54004452, -3.14917949, 0.25634965, -2.54531042], device=self.device)

		# train/freeze language encoder
		self.freeze_language_encoder = freeze_language_encoder
		if freeze_language_encoder:
			utils.freeze_params(self.model)
		else: 
			self.networks['model'] = self.model 
			self.optims.append(torch.optim.Adam(self.model.parameters(),
                        lr=lr, betas=[0.9, 0.999]))

		# losses
		if loss == "BCE":
			self.loss = nn.BCELoss()
			self.loss_no_reduction = nn.BCELoss(reduction='none')
		else:
			raise NotImplementedError

		self.PER_min, self.PER_max = PER_min, PER_max
		self.to(self.device)

	def add_mv_model(self, mv_model):
		self.mv_model = mv_model

	def forward(self, data):
		return self.forward_verbose(data)[0]

	def forward_verbose(self, data):
		meta = dict()
		states, langs = data['states'], data['language']

		# language tokens
		tokens = self.tokenizer(langs, padding="max_length")["input_ids"]
		tokens = torch.tensor(tokens).to(self.device)
		if self.freeze_language_encoder:
			with torch.no_grad():
				lang_embedding = self.model(tokens).last_hidden_state
		else:
			lang_embedding = self.model(tokens).last_hidden_state
		lang_token = self.proj_instr_encoder(lang_embedding) + self.get_type_embedding('lang')

		# state token
		states = states.to(self.device).float()
		if self.state_dim == 2:
			# Preprocess xy coordinates so scaled between [-1, 1]
			states = (states - torch.tensor([500, 0], device=self.device)) / torch.tensor([200, 400], device=self.device)
		elif self.normalize_state:
			states = states - self.state_mean
		states_enc = self.state_encoder(states)
		states_token = self.state_linear(states_enc).unsqueeze(1) + self.get_type_embedding('state')

		# observation tokens
		all_extra_tokens = []
		with torch.no_grad():
			if self.obs_encoder_id == "multimae_patch":
				obs_token = self.obs_tokenizer(obs)
				if 'extra_init_obs' in data.keys():
					all_extra_obs = data['extra_init_obs'].to(self.device)
					for extra_obs_id in range(all_extra_obs.shape[1]):
						extra_obs_token = self.obs_tokenizer(all_extra_obs[:, extra_obs_id, :, :])
						all_extra_tokens.append(extra_obs_token)
			elif self.obs_encoder_id == "mae":
				with torch.no_grad():
					# get observation embeddings
					B = data['actual_obs'].shape[0]
					viewpoint_tokens = []
					vit_obs = data['vit_obs'].to(self.device)
					data['vit_obs'] = vit_obs
					obs_token = self.mv_model.vision_model(vit_obs)
					viewpoint_tokens.append(obs_token)
				
					if 'extra_vit_obs' in data.keys() and data['extra_vit_obs'].shape[-1] > 0:
						all_extra_vit_obs = data['extra_vit_obs'].to(self.device)
						data['extra_vit_obs'] = all_extra_vit_obs
						for extra_vit_obs_id in range(all_extra_vit_obs.shape[1]):
							extra_obs_token = self.mv_model.vision_model(all_extra_vit_obs[:, extra_vit_obs_id, :, :])
							viewpoint_tokens.append(extra_obs_token)

					# concat all masked sequence
					unmasked_seq = torch.cat(viewpoint_tokens, dim=1)
					cls_tokens = self.mv_model.cls_token.expand(unmasked_seq.shape[0], -1, -1)
					obs_token = torch.cat((cls_tokens, unmasked_seq), dim=1)
			elif self.obs_encoder_id == "mvmae":
				vit_obs = data['vit_obs'].to(self.device)
				obs_token = self.vision_model(vit_obs)['last_hidden_state']
				if 'extra_vit_obs' in data.keys() and data['extra_vit_obs'].shape[-1] > 0:
					all_extra_vit_obs = data['extra_vit_obs'].to(self.device)
					for extra_vit_obs_id in range(all_extra_vit_obs.shape[1]):
						extra_obs_token = self.vision_model(all_extra_vit_obs[:, extra_vit_obs_id, :, :])['last_hidden_state']
						all_extra_tokens.append(extra_obs_token)
			elif self.obs_encoder_id in ["clip16", "clip14"]:
				vit_obs = data['vit_obs'].to(self.device)
				obs_token = self.vision_model(vit_obs)['last_hidden_state']
		if self.obs_encoder_id in ["clip16_scratch"]:
			vit_obs = data['vit_obs'].to(self.device)
			obs_token = self.vision_model(vit_obs)['last_hidden_state']
		if len(all_extra_tokens) > 0:
			extra_obs_token = torch.cat(all_extra_tokens, dim=1)
			all_obs_token = torch.cat([extra_obs_token, obs_token], dim=1)
		else:
			all_obs_token = obs_token
		featurized_obs_tokens = self.forward_obs_attn(all_obs_token)
		featurized_obs_tokens = featurized_obs_tokens + self.get_type_embedding('obs')
		
		if self.use_transformer_pos_emb:
			if self.obs_encoder_id == "mae":
				featurized_obs_tokens[:, 1:197, :] += self.transformer_pos_emb
				featurized_obs_tokens[:, 197:, :] += self.transformer_pos_emb
			else:
				raise NotImplementedError
		# concatenate all tokens
		context = torch.cat([featurized_obs_tokens, lang_token], dim=1)
		
		if self.cross_attention:
			# self-attention -> cross-attention 
			context = self.transformer.drop(context)
			for block in self.transformer.h:
			    context = block(context)
			x, transformer_meta = self.transformer.cross(states_token, context)
			meta.update(transformer_meta)
		else:
			sequence = torch.cat([context, states_token], dim=1)
			x = self.transformer.drop(sequence)
			for block in self.transformer.h:
				x = block(x)
			if self.attention_feat == "average":
				x = x.mean(dim=1)
			elif self.attention_feat == "last":
				x = x[:, -1, :]
			else:
				raise NotImplementedError
		x = x.squeeze(1)
		x = self.transformer.ln_f(x)
		pred = self.pred_head(x)
		return pred, meta

	def forward_obs_attn(self, obs_tokens):
		if self.use_mv:
			with torch.no_grad():
				return self.mv_model.forward_mae_encoder(obs_tokens, use_mask=False)
		elif self.n_obs_attn_layer > 0:
			for block in self.obs_attn:
			    obs_tokens = block(obs_tokens)
			return obs_tokens
		if self.token_dim != 768:
			obs_tokens = self.obs_linear.forward(obs_tokens)
			return obs_tokens
		else:
			return obs_tokens


	def get_type_embedding(self, name):
		if self.modality_embedding:
			return {
				'obs': self.obs_type_embedding,
				'lang': self.lang_type_embedding,
				'state': self.state_type_embedding,
			}[name]
		else:
			return 0.0

	def get_metrics(self, pred, data):
		pass

	def get_eval_metrics(self, model_out, data):
		pass

	def update(self, data, step, dataset=None, optimize=True):
		pass

	def configure_optims(self, lr, weight_decay):
		# separate out all parameters to those that will and won't experience regularizing weight decay
		decay = set()
		no_decay = set()
		whitelist_weight_modules = (torch.nn.Linear, )
		blacklist_weight_modules = (torch.nn.LayerNorm, LayerNorm, torch.nn.Embedding)
		for mn, m in self.named_modules():
			relevant_module = False
			for mod in self.networks:
				if mod in mn:
					relevant_module = True
					break
			if relevant_module is False:
				continue
			for pn, p in m.named_parameters():
			    fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
			    # random note: because named_modules and named_parameters are recursive
			    # we will see the same tensors p many many times. but doing it this way
			    # allows us to know which parent module any tensor p belongs to...
			    # print(fpn, end="|")
			    if pn.endswith('bias'):
			        # all biases will not be decayed
			        no_decay.add(fpn)
			    elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
			        # weights of whitelist modules will be weight decayed
			        decay.add(fpn)
			    elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
			        # weights of blacklist modules will NOT be weight decayed
			        no_decay.add(fpn)

		# if not already found. Not sure why the above iterates over modules instead of parameters
		for name, _ in self.named_parameters():
			relevant = False
			for net in self.networks:
				if net in name:
					relevant = True
			if not relevant:
				continue
			if name in decay or name in no_decay:
				continue
			no_decay.add(name)

		# create the pytorch optimizer object
		param_dict = {pn: p for pn, p in self.named_parameters()}
		optim_groups = [
		    {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
		    {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
		]
		# new PyTorch nightly has a new 'fused' option for AdamW that is much faster
		use_fused = 'fused' in inspect.signature(torch.optim.AdamW).parameters
		print(f"using fused AdamW: {use_fused}")
		extra_args = dict(fused=True) if use_fused else dict()
		optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=[0.9, 0.999], **extra_args)
		return optimizer

	def configure_schedulers(self, scheduler, schedule_every_step):
		self.schedule_every_step = utils.Every(schedule_every_step)
		schedulers = []
		for optim in self.optims:
			match = re.match(r'cosine\((.+),(.+)\)', scheduler)
			if match:
				T_max, eta_min = [float(g) for g in match.groups()] 
				sched = lr_scheduler.CosineAnnealingLR(optim, T_max=T_max, eta_min=eta_min)
			else:
				print("No scheduler")
				sched = None
			schedulers.append(sched)
		return schedulers

	def init_from(self, other):
		# copy parameters over
		for key in self.networks:
			if key in self.__dict__['_modules'].keys():
				utils.hard_update_params(other[key], self.__dict__['_modules'][key])
				self.__dict__['_modules'][key].to(self.device)
			elif key in self.__dict__['_parameters'].keys():
				self.__dict__['_parameters'][key].data.copy_(other[key].data)
				self.__dict__['_parameters'][key].to(self.device)
			else:
				self.__dict__[key].data.copy_(other[key].data)
				self.__dict__[key].to(self.device)
			other[key].cpu()

	def get_model(self):
		model_dict = dict()
		for key in self.networks:
			if key in self.__dict__['_modules'].keys():
				model_dict[key] = self.__dict__['_modules'][key]
			elif key in self.__dict__['_parameters'].keys():
				model_dict[key] = self.__dict__['_parameters'][key]
			else:
				model_dict[key] = self.__dict__[key]
		return model_dict