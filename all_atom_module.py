import numpy as np
import torch.nn as nn   
import torch
import torch.nn.functional as F
from Network import *
from DT import *
from local_attn import LocalSelfAttention



class LocalAttentionTransformer(nn.Module):
    def __init__(self, embed_dim, pair_dim, num_heads, window_size):
        super(LocalAttentionTransformer, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = embed_dim // num_heads

        self.attn = LocalSelfAttention(embed_dim, pair_dim, num_heads, window_size)

        self.norm = AdaLN(embed_dim, embed_dim)

        self.transition = ConditionedTransitionBlock(embed_dim, embed_dim, n=2)

        zero_init(self.attn.proj)
        zero_init(self.transition.output_proj)

    def forward(self, input):
        x, c, local_attention_pair_rep, inverse_pair_distance, local_attention_pair_mask = input

        res = x
        x = self.norm(x, c)  # Normalize the input with conditioning
        x = self.attn(x, local_attention_pair_rep,
                      inverse_pair_distance, local_attention_pair_mask)  # Apply local attention
        x = x + res  # Residual connection

        x = x + self.transition(x, c)
        return x


class AllAtomEncoder(nn.Module):
    def __init__(self, ninp = 384, n_all_atom = 128, 
                 nhead = 4, pair_dim = 128, 
                 all_atom_pair_dim = 16  , W = 32,
                 nlayers = 3):
        super(AllAtomEncoder, self).__init__()
        self.ninp = ninp
        self.n_all_atom = n_all_atom
        self.nhead = nhead
        self.pair_dim = pair_dim
        self.all_atom_pair_dim = all_atom_pair_dim
        self.W = W
        
        self.atom_embedding = nn.Embedding(256, n_all_atom)
        self.xyz_embedding = nn.Sequential(nn.Linear(3, n_all_atom, bias = False),
                                           nn.LayerNorm(n_all_atom))
            

        self.pair_concat_linear = nn.Linear(all_atom_pair_dim+1, all_atom_pair_dim)

        self.pair_mlp = nn.Sequential(
            nn.Linear(all_atom_pair_dim,all_atom_pair_dim, bias=False),
            nn.ReLU(),
            nn.Linear(all_atom_pair_dim, all_atom_pair_dim, bias = False),
            nn.ReLU(),
            nn.Linear(all_atom_pair_dim, all_atom_pair_dim, bias = False),
        )

        # self.pair_mlp2 = nn.Sequential(
        #     nn.LayerNorm(all_atom_pair_dim),
        #     TransitionLayer(all_atom_pair_dim, n=2),
        # )

        self.pair_downsample = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, all_atom_pair_dim),
        )


        self.sequence_downsample = nn.Sequential(
            nn.LayerNorm(ninp),
            nn.Linear(ninp, n_all_atom)
        )

        self.transformer=[]
        for i in range(nlayers):
            self.transformer.append(
                LocalAttentionTransformer(embed_dim=n_all_atom, 
                                          pair_dim=all_atom_pair_dim, 
                                          num_heads=nhead, 
                                          window_size=W)
            )
        self.transformer = nn.ModuleList(self.transformer)

        self.c2p = nn.Linear(n_all_atom, all_atom_pair_dim, bias=False)

    def forward(self, s, z, all_atom_index, all_atom_xyz, all_atom_res_index):
        '''
        s and z are the sequence and pair representations respectively. bs = 1 always
        all other inputs are the all atom representations that have bs >= 1
        need to add atom pair features to the all atom pair rep
        '''

        s_downsampled = self.sequence_downsample(s)#.squeeze(0)  # Remove batch dimension
        z_downsampled = self.pair_downsample(z)#.squeeze(0)  # Remove batch dimension

        # all_atom_representation has xyz
        # all_atom_conditioning has s_trunk
        all_atom_representation = self.atom_embedding(all_atom_index)    
          
        all_atom_conditioning = all_atom_representation
        all_atom_conditioning = all_atom_conditioning + s_downsampled[:, all_atom_res_index[0]]
        # print(all_atom_conditioning)
        # exit()  
        all_atom_representation = all_atom_representation + self.xyz_embedding(all_atom_xyz)

        # print(all_atom_representation)
        # print(all_atom_conditioning)
        # exit()

        local_attention_pair_rep, local_attention_pair_mask, inverse_pair_distance = \
        self.get_all_atom_pair_features(z_downsampled, all_atom_conditioning, all_atom_xyz, all_atom_res_index[0], self.W)

        #local_attention_pair_rep = torch.cat([local_attention_pair_rep, inverse_pair_distance], dim=-1)
        #local_attention_pair_rep = self.pair_concat_linear(local_attention_pair_rep)

        local_attention_pair_rep = local_attention_pair_rep + checkpoint.checkpoint(self.pair_mlp, local_attention_pair_rep, use_reentrant=False)
        #local_attention_pair_rep = local_attention_pair_rep + self.pair_mlp2(local_attention_pair_rep)



        # print("all_atom_representation shape after s_downsampled:", all_atom_representation.shape)
        # exit()
        local_attention_pair_rep = local_attention_pair_rep[None,:]
        local_attention_pair_mask = local_attention_pair_mask[None,:]


        #all_atom_conditioning = all_atom_representation
        for layer in self.transformer:
            # print(local_attention_pair_rep.shape)
            # print(inverse_pair_distance.shape)
            # exit()
            input = [all_atom_representation, 
                                            all_atom_conditioning, 
                                            local_attention_pair_rep,
                                            inverse_pair_distance,
                                            local_attention_pair_mask]
            all_atom_representation = checkpoint.checkpoint(layer,input, use_reentrant=False)

        return all_atom_representation, all_atom_conditioning, local_attention_pair_rep, inverse_pair_distance, local_attention_pair_mask

    def get_all_atom_pair_features(self,pair_rep, all_atom_conditioning,
    all_atom_xyz, all_atom_res_index, W):
        """
        Get all atom pair features for a given pair representation and residue index.
        
        Args:
            pair_rep (torch.Tensor): Pair representation tensor of shape (N, N, D).
            all_atom_res_index (torch.Tensor): Residue index array.
            W (int): Window size.
            
        Returns:
            torch.Tensor: Local attention pair representation.
        """
        # Convert all_atom_res_index to tensor
        #all_atom_res_index = torch.tensor(all_atom_res_index, dtype=torch.long)
        pair_rep = pair_rep.squeeze(0)  # Remove batch dimension if present

        # Pad the residue index
        # padded_x = F.pad(all_atom_res_index, (W, 0), mode='constant', value=0) 
        # padded_x = F.pad(padded_x, (0, W), mode='constant', value=all_atom_res_index[-1]) 
        padded_x = F.pad(all_atom_res_index, (W, 0), mode='constant', value=-1) 
        padded_x = F.pad(padded_x, (0, W), mode='constant', value=-1) 

        # print(all_atom_res_index.shape)  # Debugging line
        # exit()

        m_index = torch.arange(all_atom_res_index.shape[0], device=all_atom_res_index.device)
        # print(m_index.shape)  # Debugging line
        # print(m_index)  # Debugging line
        # exit()
        padded_m_index = F.pad(m_index, (W, 0), mode='constant', value=0)
        padded_m_index = F.pad(padded_m_index, (0, W), mode='constant', value=m_index[-1])
        padded_m_index = padded_m_index.unfold(0, size=2*W+1, step=1)

        local_attention_pair_index_i =  all_atom_res_index[:,None].expand(-1,2*W+1)
        local_attention_pair_index_j = padded_x.unfold(0, size=2*W+1, step=1)
        local_attention_pair_index = torch.stack([local_attention_pair_index_i, local_attention_pair_index_j], dim=-1)


        local_attention_pair_mask = local_attention_pair_index[:,:,1] == -1

        # Gather from pair_rep with local_attention_pair_index
        local_attention_pair_rep = pair_rep[local_attention_pair_index[:, :, 0], local_attention_pair_index[:, :, 1]]


        #add cl,cm to local_attention_pair_rep
        all_atom_conditioning=all_atom_conditioning.squeeze(0)  # Remove batch dimension if present
        all_atom_conditioning = self.c2p(all_atom_conditioning)
        local_attention_pair_rep = local_attention_pair_rep + \
                                   all_atom_conditioning[:,None] + \
                                   all_atom_conditioning[padded_m_index]


        pair_distance = torch.norm(all_atom_xyz[:,:,None] - all_atom_xyz[:,local_attention_pair_index_j], dim=-1)
        inverse_pair_distance = 1 / (pair_distance + 1e-1)  # Avoid division by zero
        inverse_pair_distance = inverse_pair_distance.unsqueeze(-1)  # Add a dimension for broadcasting

        return local_attention_pair_rep, local_attention_pair_mask, inverse_pair_distance
        # Example


class AllAtomDecoder(nn.Module):
    def __init__(self, ninp = 384, n_all_atom = 128, 
                 nhead = 4, pair_dim = 128, 
                 all_atom_pair_dim = 16  , W = 32,
                 nlayers = 4):
        super(AllAtomDecoder, self).__init__()
        self.ninp = ninp
        self.n_all_atom = n_all_atom
        self.nhead = nhead
        self.pair_dim = pair_dim
        self.all_atom_pair_dim = all_atom_pair_dim
        self.W = W
        


        self.transformer=[]
        for i in range(nlayers):
            self.transformer.append(
                LocalAttentionTransformer(embed_dim=n_all_atom, 
                                          pair_dim=all_atom_pair_dim, 
                                          num_heads=nhead, 
                                          window_size=W)
            )
        self.transformer = nn.ModuleList(self.transformer)

        self.xyz_predictor = nn.Sequential(nn.LayerNorm(n_all_atom),
                                             nn.Linear(n_all_atom, 3))

    def forward(self,   all_atom_representation, 
                        all_atom_conditioning, 
                        local_attention_pair_rep,
                        inverse_pair_distance,
                        local_attention_pair_mask):
        '''
        s and z are the sequence and pair representations respectively. bs = 1 always
        all other inputs are the all atom representations that have bs >= 1
        need to add atom pair features to the all atom pair rep
        '''




        #all_atom_conditioning = all_atom_representation
        for layer in self.transformer:
            # print(local_attention_pair_rep.shape)
            # print(inverse_pair_distance.shape)
            # exit()
            input = [all_atom_representation, 
                                            all_atom_conditioning, 
                                            local_attention_pair_rep,
                                            inverse_pair_distance,
                                            local_attention_pair_mask]
            all_atom_representation = checkpoint.checkpoint(layer,input, use_reentrant=False)

        xyz = self.xyz_predictor(all_atom_representation)

        return xyz



if __name__ == "__main__":
    seq_len = 48
    bs=8

    encoder = AllAtomEncoder().cuda()
    s = torch.randn(1, seq_len, 384).cuda()
    z = torch.randn(1, seq_len, seq_len, 128).cuda()
    all_atom_index = torch.arange(seq_len).cuda()[:,None].expand(-1,20).reshape(-1)
    all_atom_xyz = torch.randn(bs, seq_len*20, 3).cuda()
    all_atom_res_index = torch.randint(0, seq_len, (seq_len*20,)).cuda()

    all_atom_index = all_atom_index[None,:]#.expand(bs, -1)
    all_atom_res_index = all_atom_res_index[None,:].expand(bs, -1)

    print("s shape:", s.shape)  # Debugging line
    print("z shape:", z.shape)  # Debugging line
    print("all_atom_index shape:", all_atom_index.shape)  # Debugging line
    print("all_atom_xyz shape:", all_atom_xyz.shape)  # Debugging line
    print("all_atom_res_index shape:", all_atom_res_index.shape)  # Debugging line
    #exit()
    all_atom_representation, all_atom_conditioning, local_attention_pair_rep, inverse_pair_distance, local_attention_pair_mask=encoder(s, z, all_atom_index, all_atom_xyz, all_atom_res_index)
    print("all_atom_representation shape:", all_atom_representation.shape)  # Debugging line
    print("all_atom_conditioning shape:", all_atom_conditioning.shape)  # Debugging line
    print("local_attention_pair_rep shape:", local_attention_pair_rep.shape)  # Debugging line
    print("inverse_pair_distance shape:", inverse_pair_distance.shape)  # Debugging line
    print("local_attention_pair_mask shape:", local_attention_pair_mask.shape)  # Debugging line