import torch.nn as nn
import torch
from utils import random_rotation_point_cloud_torch_batch
from Network import *


class finetuned_RibonanzaNet(RibonanzaNet):
    def __init__(self, rnet_config, config, pretrained=False):
        rnet_config.dropout=config.trunk_dropout
        rnet_config.use_grad_checkpoint=True
        self.rnet_config=rnet_config
        super(finetuned_RibonanzaNet, self).__init__(rnet_config)
        if pretrained:
            self.load_state_dict(torch.load(config.pretrained_weight_path,map_location='cpu'))
        # self.ct_predictor=nn.Sequential(nn.Linear(64,256),
        #                                 nn.ReLU(),
        #                                 nn.Linear(256,64),
        #                                 nn.ReLU(),
        #                                 nn.Linear(64,1)) 
        self.dropout=nn.Dropout(0.0)

        #rnet_config.nlayers=48
        rnet_config.dropout=0.1
        self.extra_evoformer = []

        for i,layer in enumerate(range(config.folding_blocks)):

            layer=FoldingBlock(rnet_config.ninp, 
            rnet_config.nhead, 
            rnet_config.ninp*4, 
            rnet_config.pairwise_dimension,
            False, rnet_config.dim_msa)

            scale_factor=1/(i+1)**0.5
            #scale_factor=i+1
            #scale_factor=0
            recursive_linear_init(layer,scale_factor)
            zero_init(layer.attn_linear)
            zero_init(layer.sequence_transititon[2])
            zero_init(layer.pair_transition[3])
            zero_init(layer.triangle_update_out.to_out)
            zero_init(layer.triangle_update_in.to_out)

            self.extra_evoformer.append(layer)
        self.extra_evoformer=nn.ModuleList(self.extra_evoformer)

        self.recycle_sequence=nn.Sequential(nn.LayerNorm(rnet_config.ninp),
                                            nn.Linear(rnet_config.ninp,rnet_config.ninp,bias=False))
        self.recycle_pairwise=nn.Sequential(nn.LayerNorm(rnet_config.pairwise_dimension),
                                            nn.Linear(rnet_config.pairwise_dimension,rnet_config.pairwise_dimension,bias=False))

        zero_init(self.recycle_sequence[1])
        zero_init(self.recycle_pairwise[1])

        #self.layer_weights=torch.zeros(48, dtype=torch.float32)
        self.layer_weights=torch.linspace(0, 1, 48, dtype=torch.float32)
        self.layer_weights[-1]=-1e18
        self.layer_weights=nn.Parameter(self.layer_weights,requires_grad=True)


        decoder_dim=config.decoder_dim
        self.structure_module=[SimpleStructureModule(d_model=decoder_dim, nhead=config.decoder_nhead, 
                 s_model=rnet_config.ninp,
                 dim_feedforward=decoder_dim*2, pairwise_dimension=rnet_config.pairwise_dimension, dropout=0.0) for i in range(config.decoder_num_layers)]
        self.structure_module=nn.ModuleList(self.structure_module)

        for i,layer in enumerate(self.structure_module):
            scale_factor=1/(i+1)**0.5
            #scale_factor=i+1
            #scale_factor=0
            recursive_linear_init(layer,scale_factor)
            zero_init(layer.attn_linear)
            zero_init(layer.conditioned_transiton.output_proj)


        self.xyz_embedder=nn.Linear(3,rnet_config.ninp)
        self.xyz_norm=nn.LayerNorm(rnet_config.ninp)
        self.xyz_predictor=nn.Sequential(nn.LayerNorm(decoder_dim),
                                         nn.Linear(decoder_dim,3))
                                            
        


        self.distogram_predictor=nn.Sequential(nn.LayerNorm(rnet_config.pairwise_dimension),
                                                nn.Linear(rnet_config.pairwise_dimension,40))

        self.time_embedder=SinusoidalPosEmb(rnet_config.ninp)

        self.concat_linear=nn.Sequential(nn.Linear(rnet_config.ninp+decoder_dim+3,decoder_dim),nn.LayerNorm(decoder_dim))

        self.time_mlp1=nn.Sequential(
                                     nn.Linear(rnet_config.ninp,rnet_config.ninp*2),
                                     nn.ReLU(),  
                                     nn.Linear(rnet_config.ninp*2,rnet_config.ninp*1))
        self.time_norm1=nn.LayerNorm(rnet_config.ninp)

        self.time_mlp2=nn.Sequential(
                                     nn.Linear(rnet_config.ninp,rnet_config.ninp*2),
                                     nn.ReLU(),  
                                     nn.Linear(rnet_config.ninp*2,rnet_config.ninp*1))
        self.time_norm2=nn.LayerNorm(rnet_config.ninp)

        self.time_mlp3=nn.Sequential(
                                     nn.Linear(rnet_config.ninp,rnet_config.ninp*2),
                                     nn.ReLU(),  
                                     nn.Linear(rnet_config.ninp*2,rnet_config.ninp*1))
        self.time_norm3=nn.LayerNorm(rnet_config.ninp)

        self.tgt_norm=nn.LayerNorm(rnet_config.ninp)

        self.distance2pairwise=nn.Linear(1,rnet_config.pairwise_dimension,bias=False)

        # self.pair_mlp1=nn.Sequential(nn.LayerNorm(rnet_config.pairwise_dimension),
        #                             nn.Linear(rnet_config.pairwise_dimension,rnet_config.pairwise_dimension*2),
        #                             nn.ReLU(),
        #                             nn.Linear(rnet_config.pairwise_dimension*2,rnet_config.pairwise_dimension))

        # self.pair_mlp2=nn.Sequential(nn.LayerNorm(rnet_config.pairwise_dimension),
        #                             nn.Linear(rnet_config.pairwise_dimension,rnet_config.pairwise_dimension*2),
        #                             nn.ReLU(),
        #                             nn.Linear(rnet_config.pairwise_dimension*2,rnet_config.pairwise_dimension))

        self.pair_distance_mlp=nn.Sequential(nn.Linear(4,8),
                                                nn.ReLU(),
                                                nn.Linear(8,4))
        self.pair_distance_mlp2=nn.Sequential(  nn.LayerNorm(4),
                                                nn.Linear(4,8),
                                                nn.ReLU(),
                                                nn.Linear(8,4))

        self.pair_vector_linear=nn.Linear(3,rnet_config.pairwise_dimension,bias=False)

        #hyperparameters for diffusion
        self.n_times = config.n_times

        #self.model = model
        
        # define linear variance schedule(betas)
        beta_1, beta_T = config.beta_min, config.beta_max
        betas = torch.linspace(start=beta_1, end=beta_T, steps=config.n_times)#.to(device) # follows DDPM paper
        self.sqrt_betas = torch.sqrt(betas)
                                     
        # define alpha for forward diffusion kernel
        self.alphas = 1 - betas
        self.sqrt_alphas = torch.sqrt(self.alphas)
        alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1-alpha_bars)
        self.sqrt_alpha_bars = torch.sqrt(alpha_bars)

        self.data_std=config.data_std

        self.adaptor=nn.Linear(rnet_config.ninp,config.decoder_dim,bias=False)

    def custom(self, module):
        def custom_forward(*inputs):
            inputs = module(*inputs)
            return inputs
        return custom_forward
    
    def embed_pair_distance(self,inputs):
        pairwise_features,xyz=inputs
        vector_matrix=(xyz[:,None,:,:]-xyz[:,:,None,:])#*self.data_std

        distance_matrix=(vector_matrix**2).sum(-1)
        distance_matrix=1/(1+distance_matrix)
        distance_matrix=distance_matrix[:,:,:,None]
        # pairwise_features=pairwise_features+\
        #                   self.distance2pairwise(distance_matrix)+\
        #                   self.pair_vector_linear(vector_matrix)

        # pairwise_features+=self.pair_mlp1(pairwise_features)
        # pairwise_features+=self.pair_mlp2(pairwise_features)
        distance_features=torch.cat([vector_matrix,distance_matrix],-1)
        distance_features=distance_features+self.pair_distance_mlp(distance_features)
        distance_features=distance_features+self.pair_distance_mlp2(distance_features)
        return distance_features



    def get_conditioning(self,src,cycles,trunk_grad=False):
        
        #print(f'Get conditioning with trunk_grad={trunk_grad} and cycles={cycles}')
        with torch.set_grad_enabled(trunk_grad):
            all_sequence_features, all_pairwise_features=self.get_embeddings(src, torch.ones_like(src).long().to(src.device))


        B,L =all_sequence_features.shape[1],all_sequence_features.shape[2]
        s_hat=torch.zeros(B,L,self.rnet_config.ninp).to(src.device)
        z_hat=torch.zeros(B,L,L,self.rnet_config.pairwise_dimension).to(src.device)

        for c in range(1,cycles+1):
            #print(f'Cycle {c}/{cycles}')
            with torch.set_grad_enabled(self.training and c==cycles):
            #with torch.no_grad():
                sequence_features=all_sequence_features*self.layer_weights.softmax(0)[:,None,None,None]
                pairwise_features=all_pairwise_features*self.layer_weights.softmax(0)[:,None,None,None,None]
                #print(self.layer_weights.softmax(0))
                sequence_features_init=sequence_features.sum(0)
                pairwise_features_init=pairwise_features.sum(0)

                sequence_features=sequence_features_init+self.recycle_sequence(s_hat)
                pairwise_features=pairwise_features_init+self.recycle_pairwise(z_hat)

                mask = torch.ones_like(src).long().to(src.device)
                if c==cycles:
                    self.extra_evoformer.requires_grad_(True)
                else:
                    self.extra_evoformer.requires_grad_(False)

                #print(f'running cycle {c} with trunk_grad={trunk_grad} and cycles={cycles}')


                for layer in self.extra_evoformer:
                    #with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    if c==cycles:
                        sequence_features,pairwise_features=checkpoint.checkpoint(layer, 
                        [sequence_features, pairwise_features, mask , False],
                        use_reentrant=False)
                    else:
                        sequence_features,pairwise_features=layer([sequence_features, pairwise_features, mask.to(src.device), False])
                    #sequence_features,pairwise_features=layer([sequence_features, pairwise_features, torch.ones_like(src).long().to(src.device), False])
                s_hat=sequence_features
                z_hat=pairwise_features

                # if c!=cycles:
                #     s_hat=s_hat.detach()
                #     z_hat=z_hat.detach()

        return sequence_features,pairwise_features

    def get_decoder_features(self, sequence_features, pairwise_features, xyz, t):
        decoder_batch_size=xyz.shape[0]
        sequence_features=sequence_features.repeat(decoder_batch_size,1,1)
        


        time_embed=self.time_norm3(self.time_mlp3(self.time_embedder(t).unsqueeze(1)))

        xyz=self.xyz_norm(self.xyz_embedder(xyz))
        


        tgt=self.tgt_norm(time_embed+sequence_features+xyz)
        tgt=self.time_norm1(tgt+self.time_mlp1(tgt))
        tgt=self.time_norm2(tgt+self.time_mlp2(tgt))

        
        return tgt, pairwise_features

    def forward(self,src,xyz,t, trunk_grad, N_cycle=4):
        
        sequence_features, pairwise_features=self.get_conditioning(src,N_cycle,trunk_grad)

        distogram=self.distogram_predictor(pairwise_features)

        sequence_features, pairwise_features=self.get_decoder_features(sequence_features, pairwise_features, xyz, t)

        tgt=self.adaptor(sequence_features)

        # print(tgt.shape)
        # print(sequence_features.shape)
        # exit()

        for layer in self.structure_module:
            #tgt=layer([tgt, sequence_features,pairwise_features,xyz,None])
            tgt=checkpoint.checkpoint(self.custom(layer),
            [tgt, sequence_features,pairwise_features,None],
            use_reentrant=False)
            # xyz=xyz+self.xyz_predictor(sequence_features).squeeze(0)
            # xyzs.append(xyz)
            #print(sequence_features.shape)
        
        xyz=self.xyz_predictor(tgt).squeeze(0)
        #.squeeze(0)

        return xyz, distogram
    

    def denoise(self,sequence_features,pairwise_features,xyz,t):
        
        sequence_features, pairwise_features=self.get_decoder_features(sequence_features, pairwise_features, xyz, t)

        tgt=self.adaptor(sequence_features)

        for layer in self.structure_module:
            tgt=layer([tgt, sequence_features,pairwise_features,None])
            # xyz=xyz+self.xyz_predictor(sequence_features).squeeze(0)
            # xyzs.append(xyz)
            #print(sequence_features.shape)
        xyz=self.xyz_predictor(tgt).squeeze(0)
        # print(xyz.shape)
        # exit()
        return xyz


    def extract(self, a, t, x_shape):
        """
            from lucidrains' implementation
                https://github.com/lucidrains/denoising-diffusion-pytorch/blob/beb2f2d8dd9b4f2bd5be4719f37082fe061ee450/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L376
        """
        b, *_ = t.shape
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))
    
    def scale_to_minus_one_to_one(self, x):
        # according to the DDPMs paper, normalization seems to be crucial to train reverse process network
        return x * 2 - 1
    
    def reverse_scale_to_zero_to_one(self, x):
        return (x + 1) * 0.5
    
    def make_noisy(self, x_zeros, t): 
        # assume we get raw data, so center and scale by 35
        x_zeros = x_zeros - torch.nanmean(x_zeros,1,keepdim=True)
        x_zeros = x_zeros/self.data_std
        #rotate randomly
        x_zeros = random_rotation_point_cloud_torch_batch(x_zeros)


        # perturb x_0 into x_t (i.e., take x_0 samples into forward diffusion kernels)
        epsilon = torch.randn_like(x_zeros).to(x_zeros.device)
        
        sqrt_alpha_bar = self.extract(self.sqrt_alpha_bars.to(x_zeros.device), t, x_zeros.shape)
        sqrt_one_minus_alpha_bar = self.extract(self.sqrt_one_minus_alpha_bars.to(x_zeros.device), t, x_zeros.shape)
        
        # Let's make noisy sample!: i.e., Forward process with fixed variance schedule
        #      i.e., sqrt(alpha_bar_t) * x_zero + sqrt(1-alpha_bar_t) * epsilon
        noisy_sample = x_zeros * sqrt_alpha_bar + epsilon * sqrt_one_minus_alpha_bar
    
        return noisy_sample.detach(), epsilon
    
    
    # def forward(self, x_zeros):
    #     x_zeros = self.scale_to_minus_one_to_one(x_zeros)
        
    #     B, _, _, _ = x_zeros.shape
        
    #     # (1) randomly choose diffusion time-step
    #     t = torch.randint(low=0, high=self.n_times, size=(B,)).long().to(x_zeros.device)
        
    #     # (2) forward diffusion process: perturb x_zeros with fixed variance schedule
    #     perturbed_images, epsilon = self.make_noisy(x_zeros, t)
        
    #     # (3) predict epsilon(noise) given perturbed data at diffusion-timestep t.
    #     pred_epsilon = self.model(perturbed_images, t)
        
    #     return perturbed_images, epsilon, pred_epsilon
    
    
    def denoise_at_t(self, x_t, sequence_features, pairwise_features, timestep, t):
        B, _, _ = x_t.shape
        if t > 1:
            z = torch.randn_like(x_t).to(sequence_features.device)
        else:
            z = torch.zeros_like(x_t).to(sequence_features.device)
        
        # at inference, we use predicted noise(epsilon) to restore perturbed data sample.
        epsilon_pred = self.denoise(sequence_features, pairwise_features, x_t, timestep)
        
        alpha = self.extract(self.alphas.to(x_t.device), timestep, x_t.shape)
        sqrt_alpha = self.extract(self.sqrt_alphas.to(x_t.device), timestep, x_t.shape)
        sqrt_one_minus_alpha_bar = self.extract(self.sqrt_one_minus_alpha_bars.to(x_t.device), timestep, x_t.shape)
        sqrt_beta = self.extract(self.sqrt_betas.to(x_t.device), timestep, x_t.shape)
        
        # denoise at time t, utilizing predicted noise
        x_t_minus_1 = 1 / sqrt_alpha * (x_t - (1-alpha)/sqrt_one_minus_alpha_bar*epsilon_pred) + sqrt_beta*z
        
        return x_t_minus_1#.clamp(-1., 1)
                
    def sample(self, src, N, N_cycle=1):
        device = src.device
        x_t = torch.randn((N, src.shape[1], 3)).to(device)

        # Get conditioning
        with torch.no_grad():
            sequence_features, pairwise_features=self.get_conditioning(src,N_cycle)

        distogram = self.distogram_predictor(pairwise_features).squeeze()
        distogram = distogram.squeeze()[:, :, 2:40].softmax(-1) * torch.arange(2, 40).float().to(device)
        distogram = distogram.sum(-1)

        for t in range(self.n_times-1, -1, -1):
            timestep = torch.tensor([t]).repeat_interleave(N, dim=0).long().to(src.device)
            x_t = self.denoise_at_t(x_t, sequence_features, pairwise_features, timestep, t)
        
        # denormalize x_0 into 0 ~ 1 ranged values.
        #x_0 = self.reverse_scale_to_zero_to_one(x_t)
        x_0 = x_t * self.data_std
        return x_0, distogram

    def sample_heun(self, src, N, num_steps=None, eta=0.0):
        """
        Heun's method sampler with optional fewer steps (coarse stepping).
        
        Args:
            src (torch.Tensor): Input sequence tensor.
            N (int): Number of samples to generate.
            num_steps (int, optional): Number of timesteps to sample with. If None, uses full DDPM schedule.
        """
        device = src.device
        x_t = torch.randn((N, src.shape[1], 3)).to(device)

        # Get conditioning
        # Get conditioning
        with torch.no_grad():
            sequence_features, pairwise_features=self.get_conditioning(src)

        distogram = self.distogram_predictor(pairwise_features).squeeze()
        distogram = distogram.squeeze()[:, :, 2:40].softmax(-1) * torch.arange(2, 40).float().to(device)
        distogram = distogram.sum(-1)

        if num_steps is None:
            timesteps = list(range(self.n_times - 1, 0, -1))
        else:
            timesteps = torch.linspace(self.n_times - 1, 1, steps=num_steps, dtype=torch.long).tolist()
        # print(timesteps)
        # exit()
        for i in range(len(timesteps)):
            t = int(timesteps[i])
            t_next = int(timesteps[i + 1]) if i + 1 < len(timesteps) else 0
            t_next_next = int(timesteps[i + 1]) if i + 1 < len(timesteps) else 0

            t_curr = torch.full((N,), t, dtype=torch.long).to(device)
            t_next_tensor = torch.full((N,), t_next, dtype=torch.long).to(device)
            t_next_next_tensor = torch.full((N,), t_next_next, dtype=torch.long).to(device)

            alpha_bar_t = self.extract(self.sqrt_alpha_bars.to(device), t_curr, x_t.shape) ** 2
            alpha_bar_next = self.extract(self.sqrt_alpha_bars.to(device), t_next_tensor, x_t.shape) ** 2
            alpha_bar_next_next = self.extract(self.sqrt_alpha_bars.to(device), t_next_next_tensor, x_t.shape) ** 2

            #Predict noise at x_t
            eps1 = self.denoise(sequence_features, pairwise_features, x_t, t_curr)

            # # Predict x_0
            x_0 = (x_t - eps1 * torch.sqrt(1 - alpha_bar_t)) / torch.sqrt(alpha_bar_t)

            #grad1 = eps1 * torch.sqrt(1 - alpha_bar_t)
            # Euler step
            x_t_euler = torch.sqrt(alpha_bar_next) * x_0 + torch.sqrt(1 - alpha_bar_next) * eps1
            #x_t = x_t_euler
            # # Predict noise at x_{t-1}

            #print(torch.square(x_t-x_t_euler).mean())
            #x_t_euler = batched_svd_align(x_t_euler, x_t)
            # print(torch.square(x_t-x_t_euler).mean())
            # exit()
            eps2 = self.denoise(sequence_features, pairwise_features, x_t_euler, t_next_tensor)

            # Predict x_0
            #x_0 = (x_t - eps1 * torch.sqrt(1 - alpha_bar_t)) / torch.sqrt(alpha_bar_t)

            x_0_next = (x_t_euler - eps2 * torch.sqrt(1 - alpha_bar_next_next)) / torch.sqrt(alpha_bar_next_next)
            x_t_next = torch.sqrt(alpha_bar_next) * x_0_next + torch.sqrt(1 - alpha_bar_next) * eps2

            x_t = 0.5 * (x_t_euler + x_t_next)

            # sigma_t = eta * torch.sqrt((1 - alpha_bar_next) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_next))
            # # print(sigma_t)
            # # exit()

            # #x_0_heun = 0.5 * (x_0+  batched_svd_align(x_0_next, x_0))
            # eps_avg = 0.5 * (eps1 + eps2)
            # x_0_avg = 0.5 * (x_0 + x_0_next)
            # x_t = torch.sqrt(alpha_bar_next) * x_0_avg + \
            #       torch.sqrt(1 - alpha_bar_next-sigma_t**2) * eps_avg + \
            #       sigma_t * torch.randn_like(x_t) #* torch.sqrt(1 - alpha_bar_next)


            # # # Heun step (second-order correction)
            # eps_avg = 0.5 * (eps1 + eps2)
            # x_0 = (x_t - eps_avg * torch.sqrt(1 - alpha_bar_t)) / torch.sqrt(alpha_bar_t)
            # x_t = torch.sqrt(alpha_bar_next) * x_0 + torch.sqrt(1 - alpha_bar_next) * eps_avg

            # x_t = (x_t - eps1 * torch.sqrt(1 - alpha_bar_t)) / torch.sqrt(alpha_bar_t) * torch.sqrt(alpha_bar_next) + \
            #     torch.sqrt(1 - alpha_bar_next) * eps1

            
            # # Predict noise at x_t
            # eps1 = self.denoise(sequence_features, pairwise_features, x_t, t_curr)


            # # Euler step
            # scale_factor = torch.sqrt(alpha_bar_next)/torch.sqrt(alpha_bar_next)
            # step_size = (torch.sqrt((1 - alpha_bar_next)*alpha_bar_next) / torch.sqrt(alpha_bar_t) + torch.sqrt(1 - alpha_bar_next))

            #x_t_euler = x_t * scale_factor - eps1 * step_size

            #x_t = x_t_euler

            #if stochastic:
            #sigma_t = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t/alpha_prev))
            # sigma_t = eta * torch.sqrt((1 - alpha_bar_next) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_next))
            # x_t_euler = x_t * scale_factor - eps1 * step_size + torch.randn_like(x_t) * sigma_t

            # #grad1 = eps1 * step_size / scale_factor
            # x_t_euler = batched_svd_align(x_t_euler, x_t)
            # eps2 = self.denoise(sequence_features, pairwise_features, x_t_euler, t_next_tensor)
            # eps_avg = 0.5 * (eps1 + eps2)
            # x_t = x_t * scale_factor - eps_avg * step_size + torch.randn_like(x_t) * sigma_t

        x_0 = x_t * self.data_std
        return x_0, distogram       



    def sample_euler(self, src, N, num_steps=None, eta=0.0, N_cycle=4):
        """
        Heun's method sampler with optional fewer steps (coarse stepping).
        
        Args:
            src (torch.Tensor): Input sequence tensor.
            N (int): Number of samples to generate.
            num_steps (int, optional): Number of timesteps to sample with. If None, uses full DDPM schedule.
        """
        device = src.device
        x_t = torch.randn((N, src.shape[1], 3)).to(device)

        # Get conditioning
        with torch.no_grad():
            sequence_features, pairwise_features=self.get_conditioning(src,N_cycle)

        distogram = self.distogram_predictor(pairwise_features).squeeze()
        distogram = distogram.squeeze()[:, :, 2:40].softmax(-1) * torch.arange(2, 40).float().to(device)
        distogram = distogram.sum(-1)

        if num_steps is None:
            timesteps = list(range(self.n_times - 1, 0, -1))
        else:
            timesteps = torch.linspace(self.n_times - 1, 1, steps=num_steps, dtype=torch.long).tolist()

        for i in range(len(timesteps)):
            t = int(timesteps[i])
            t_next = int(timesteps[i + 1]) if i + 1 < len(timesteps) else 0

            t_curr = torch.full((N,), t, dtype=torch.long).to(device)
            t_next_tensor = torch.full((N,), t_next, dtype=torch.long).to(device)

            alpha_bar_t = self.extract(self.sqrt_alpha_bars.to(device), t_curr, x_t.shape) ** 2
            alpha_bar_next = self.extract(self.sqrt_alpha_bars.to(device), t_next_tensor, x_t.shape) ** 2

            # Predict noise at x_t
            eps1 = self.denoise(sequence_features, pairwise_features, x_t, t_curr)


            # Euler step
            scale_factor = torch.sqrt(alpha_bar_t)/torch.sqrt(alpha_bar_next)
            step_size = (torch.sqrt((1 - alpha_bar_next)*alpha_bar_next) / torch.sqrt(alpha_bar_t) + torch.sqrt(1 - alpha_bar_next))

            #x_t_euler = x_t * scale_factor - eps1 * step_size

            #x_t = x_t_euler

            #if stochastic:
            #sigma_t = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t/alpha_prev))
            sigma_t = eta * torch.sqrt((1 - alpha_bar_next) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_next))
            sqrt_beta = self.extract(self.sqrt_betas.to(t_curr.device), t_curr, x_t.shape)
            z = torch.randn_like(x_t) if i + 1 < len(timesteps) else torch.zeros_like(x_t)
            x_t = x_t * scale_factor - eps1 * step_size +  z * sqrt_beta * eta




        x_0 = x_t * self.data_std
        return x_0, distogram        