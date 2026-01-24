import torch.nn as nn
import torch
import numpy as np
import math

from models.modules import ParityBackbone, SynapseUNET, Squeeze, SuperLinear, LearnableFourierPositionalEncoding, MultiLearnableFourierPositionalEncoding, CustomRotationalEmbedding, CustomRotationalEmbedding1D, ShallowWide
from models.resnet import prepare_resnet_backbone
from models.utils import compute_normalized_entropy

from models.constants import (
    VALID_NEURON_SELECT_TYPES,
    VALID_BACKBONE_TYPES,
    VALID_POSITIONAL_EMBEDDING_TYPES
)

class ContinuousThoughtMachine(nn.Module):
                              

    def __init__(self,
                 iterations,
                 d_model,
                 d_input,
                 heads,
                 n_synch_out,
                 n_synch_action,
                 synapse_depth,
                 memory_length,
                 deep_nlms,
                 memory_hidden_dims,
                 do_layernorm_nlm,
                 backbone_type,
                 positional_embedding_type,
                 out_dims,
                 prediction_reshaper=[-1],
                 dropout=0,
                 dropout_nlm=None,
                 neuron_select_type='random-pairing',  
                 n_random_pairing_self=0,
                 ):
        super(ContinuousThoughtMachine, self).__init__()

        # --- Core Parameters ---
        self.iterations = iterations
        self.d_model = d_model
        self.d_input = d_input
        self.memory_length = memory_length
        self.prediction_reshaper = prediction_reshaper
        self.n_synch_out = n_synch_out
        self.n_synch_action = n_synch_action
        self.backbone_type = backbone_type
        self.out_dims = out_dims
        self.positional_embedding_type = positional_embedding_type
        self.neuron_select_type = neuron_select_type
        self.memory_length = memory_length
        dropout_nlm = dropout if dropout_nlm is None else dropout_nlm

        # --- Assertions ---
        self.verify_args()

        # --- Input Processing  ---
        d_backbone = self.get_d_backbone()
        self.set_initial_rgb()
        self.set_backbone()
        self.positional_embedding = self.get_positional_embedding(d_backbone)
        self.kv_proj = nn.Sequential(nn.LazyLinear(self.d_input), nn.LayerNorm(self.d_input)) if heads else None
        self.q_proj = nn.LazyLinear(self.d_input) if heads else None
        self.attention = nn.MultiheadAttention(self.d_input, heads, dropout, batch_first=True) if heads else None
        
        # --- Core CTM Modules ---
        self.synapses = self.get_synapses(synapse_depth, d_model, dropout)
        self.trace_processor = self.get_neuron_level_models(deep_nlms, do_layernorm_nlm, memory_length, memory_hidden_dims, d_model, dropout_nlm)

        #  --- Start States ---
        self.register_parameter('start_activated_state', nn.Parameter(torch.zeros((d_model)).uniform_(-math.sqrt(1/(d_model)), math.sqrt(1/(d_model)))))
        self.register_parameter('start_trace', nn.Parameter(torch.zeros((d_model, memory_length)).uniform_(-math.sqrt(1/(d_model+memory_length)), math.sqrt(1/(d_model+memory_length)))))

        # --- Synchronisation ---
        self.neuron_select_type_out, self.neuron_select_type_action = self.get_neuron_select_type()
        self.synch_representation_size_action = self.calculate_synch_representation_size(self.n_synch_action)
        self.synch_representation_size_out = self.calculate_synch_representation_size(self.n_synch_out)
        
        for synch_type, size in (('action', self.synch_representation_size_action), ('out', self.synch_representation_size_out)):
            print(f"Synch representation size {synch_type}: {size}")
        if self.synch_representation_size_action:  # if not zero
            self.set_synchronisation_parameters('action', self.n_synch_action, n_random_pairing_self)
        self.set_synchronisation_parameters('out', self.n_synch_out, n_random_pairing_self)

        # --- Output Procesing ---
        self.output_projector = nn.Sequential(nn.LazyLinear(self.out_dims))


    def compute_synchronisation(self, activated_state, decay_alpha, decay_beta, r, synch_type):
        

        if synch_type == 'action': # Get action parameters
            n_synch = self.n_synch_action
            neuron_indices_left = self.action_neuron_indices_left
            neuron_indices_right = self.action_neuron_indices_right
        elif synch_type == 'out': # Get input parameters
            n_synch = self.n_synch_out
            neuron_indices_left = self.out_neuron_indices_left
            neuron_indices_right = self.out_neuron_indices_right
        
        if self.neuron_select_type in ('first-last', 'random'):
            
            if self.neuron_select_type == 'first-last':
                if synch_type == 'action': # Use last n_synch neurons for action
                    selected_left = selected_right = activated_state[:, -n_synch:]
                elif synch_type == 'out': # Use first n_synch neurons for out
                    selected_left = selected_right = activated_state[:, :n_synch]
            else: # Use the randomly selected neurons
                selected_left = activated_state[:, neuron_indices_left]
                selected_right = activated_state[:, neuron_indices_right]
            
            # Compute outer product of selected neurons
            outer = selected_left.unsqueeze(2) * selected_right.unsqueeze(1)
            
            i, j = torch.triu_indices(n_synch, n_synch)
            pairwise_product = outer[:, i, j]
            
        elif self.neuron_select_type == 'random-pairing':
            
            left = activated_state[:, neuron_indices_left]
            right = activated_state[:, neuron_indices_right]
            pairwise_product = left * right
        else:
            raise ValueError("Invalid neuron selection type")
        
        

        if decay_alpha is None or decay_beta is None:
            decay_alpha = pairwise_product
            decay_beta = torch.ones_like(pairwise_product)
        else:
            decay_alpha = r * decay_alpha + pairwise_product
            decay_beta = r * decay_beta + 1
        
        synchronisation = decay_alpha / (torch.sqrt(decay_beta))
        return synchronisation, decay_alpha, decay_beta

    def compute_features(self, x):
 
        self.kv_features = x
        kv = self.kv_proj(self.kv_features)
        
        return kv

    def compute_certainty(self, current_prediction):

        B = current_prediction.size(0)
        reshaped_pred = current_prediction.reshape([B] + self.prediction_reshaper)
        ne = compute_normalized_entropy(reshaped_pred)
        current_certainty = torch.stack((ne, 1-ne), -1)
        return current_certainty

  

    def set_initial_rgb(self):

        if 'resnet' in self.backbone_type:
            self.initial_rgb = nn.LazyConv2d(3, 1, 1) 
        else:
            self.initial_rgb = nn.Identity()

    def get_d_backbone(self):
        
        if self.backbone_type == 'shallow-wide':
            return 2048
        elif self.backbone_type == 'parity_backbone':
            return self.d_input
        elif 'resnet' in self.backbone_type:
            if '18' in self.backbone_type or '34' in self.backbone_type: 
                if self.backbone_type.split('-')[1]=='1': return 64
                elif self.backbone_type.split('-')[1]=='2': return 128
                elif self.backbone_type.split('-')[1]=='3': return 256
                elif self.backbone_type.split('-')[1]=='4': return 512
                else:
                    raise NotImplementedError
            else:
                if self.backbone_type.split('-')[1]=='1': return 256
                elif self.backbone_type.split('-')[1]=='2': return 512
                elif self.backbone_type.split('-')[1]=='3': return 1024
                elif self.backbone_type.split('-')[1]=='4': return 2048
                else:
                    raise NotImplementedError
        elif self.backbone_type == 'none':
            return None
        else:
            raise ValueError(f"Invalid backbone_type: {self.backbone_type}")

    def set_backbone(self):
        """
        Set the backbone module based on the specified type.
        """
        if self.backbone_type == 'shallow-wide':
            self.backbone = ShallowWide()
        elif self.backbone_type == 'parity_backbone':
            d_backbone = self.get_d_backbone()
            self.backbone = ParityBackbone(n_embeddings=2, d_embedding=d_backbone)
        elif 'resnet' in self.backbone_type:
            self.backbone = prepare_resnet_backbone(self.backbone_type)
        elif self.backbone_type == 'none':
            self.backbone = nn.Identity()
        else:
            raise ValueError(f"Invalid backbone_type: {self.backbone_type}")

    def get_positional_embedding(self, d_backbone):
        """
        """
        if self.positional_embedding_type == 'learnable-fourier':
            return LearnableFourierPositionalEncoding(d_backbone, gamma=1 / 2.5)
        elif self.positional_embedding_type == 'multi-learnable-fourier':
            return MultiLearnableFourierPositionalEncoding(d_backbone)
        elif self.positional_embedding_type == 'custom-rotational':
            return CustomRotationalEmbedding(d_backbone)
        elif self.positional_embedding_type == 'custom-rotational-1d':
            return CustomRotationalEmbedding1D(d_backbone)
        elif self.positional_embedding_type == 'none':
            return lambda x: 0  # Default no-op
        else:
            raise ValueError(f"Invalid positional_embedding_type: {self.positional_embedding_type}")

    def get_neuron_level_models(self, deep_nlms, do_layernorm_nlm, memory_length, memory_hidden_dims, d_model, dropout):

        if deep_nlms:
            return nn.Sequential(
                nn.Sequential(
                    SuperLinear(in_dims=memory_length, out_dims=2 * memory_hidden_dims, N=d_model,
                                do_norm=do_layernorm_nlm, dropout=dropout),
                    nn.GLU(),
                    SuperLinear(in_dims=memory_hidden_dims, out_dims=2, N=d_model,
                                do_norm=do_layernorm_nlm, dropout=dropout),
                    nn.GLU(),
                    Squeeze(-1)
                )
            )
        else:
            return nn.Sequential(
                nn.Sequential(
                    SuperLinear(in_dims=memory_length, out_dims=2, N=d_model,
                                do_norm=do_layernorm_nlm, dropout=dropout),
                    nn.GLU(),
                    Squeeze(-1)
                )
            )

    def get_synapses(self, synapse_depth, d_model, dropout):

        if synapse_depth == 1:
            return nn.Sequential(
                nn.Dropout(dropout),
                nn.LazyLinear(d_model * 2),
                nn.GLU(),
                nn.LayerNorm(d_model)
            )
        else:
            return SynapseUNET(d_model, synapse_depth, 16, dropout)  # 
    def set_synchronisation_parameters(self, synch_type: str, n_synch: int, n_random_pairing_self: int = 0):

            assert synch_type in ('out', 'action'), f"Invalid synch_type: {synch_type}"
            left, right = self.initialize_left_right_neurons(synch_type, self.d_model, n_synch, n_random_pairing_self)
            synch_representation_size = self.synch_representation_size_action if synch_type == 'action' else self.synch_representation_size_out
            self.register_buffer(f'{synch_type}_neuron_indices_left', left)
            self.register_buffer(f'{synch_type}_neuron_indices_right', right)
            self.register_parameter(f'decay_params_{synch_type}', nn.Parameter(torch.zeros(synch_representation_size), requires_grad=True))

    def initialize_left_right_neurons(self, synch_type, d_model, n_synch, n_random_pairing_self=0):

        if self.neuron_select_type=='first-last':
            if synch_type == 'out':
                neuron_indices_left = neuron_indices_right = torch.arange(0, n_synch)
            elif synch_type == 'action':
                neuron_indices_left = neuron_indices_right = torch.arange(d_model-n_synch, d_model)

        elif self.neuron_select_type=='random':
            neuron_indices_left = torch.from_numpy(np.random.choice(np.arange(d_model), size=n_synch))
            neuron_indices_right = torch.from_numpy(np.random.choice(np.arange(d_model), size=n_synch))

        elif self.neuron_select_type=='random-pairing':
            assert n_synch > n_random_pairing_self, f"Need at least {n_random_pairing_self} pairs for {self.neuron_select_type}"
            neuron_indices_left = torch.from_numpy(np.random.choice(np.arange(d_model), size=n_synch))
            neuron_indices_right = torch.concatenate((neuron_indices_left[:n_random_pairing_self], torch.from_numpy(np.random.choice(np.arange(d_model), size=n_synch-n_random_pairing_self))))

        device = self.start_activated_state.device
        return neuron_indices_left.to(device), neuron_indices_right.to(device)

    def get_neuron_select_type(self):
 
        print(f"Using neuron select type: {self.neuron_select_type}")
        if self.neuron_select_type == 'first-last':
            neuron_select_type_out, neuron_select_type_action = 'first', 'last'
        elif self.neuron_select_type in ('random', 'random-pairing'):
            neuron_select_type_out = neuron_select_type_action = self.neuron_select_type
        else:
            raise ValueError(f"Invalid neuron selection type: {self.neuron_select_type}")
        return neuron_select_type_out, neuron_select_type_action

    # --- Utilty Methods ---

    def verify_args(self):

        assert self.neuron_select_type in VALID_NEURON_SELECT_TYPES, \
            f"Invalid neuron selection type: {self.neuron_select_type}"
        
        assert self.backbone_type in VALID_BACKBONE_TYPES + ['none'], \
            f"Invalid backbone_type: {self.backbone_type}"
        
        assert self.positional_embedding_type in VALID_POSITIONAL_EMBEDDING_TYPES + ['none'], \
            f"Invalid positional_embedding_type: {self.positional_embedding_type}"
        
        if self.neuron_select_type == 'first-last':
            assert self.d_model >= (self.n_synch_out + self.n_synch_action), \
                "d_model must be >= n_synch_out + n_synch_action for neuron subsets"

        if self.backbone_type=='none' and self.positional_embedding_type!='none':
            raise AssertionError("There should be no positional embedding if there is no backbone.")

    def calculate_synch_representation_size(self, n_synch):
        """
        Calculate the size of the synchronisation representation based on neuron selection type.
        """
        if self.neuron_select_type == 'random-pairing':
            synch_representation_size = n_synch
        elif self.neuron_select_type in ('first-last', 'random'):
            synch_representation_size = (n_synch * (n_synch + 1)) // 2
        else:
            raise ValueError(f"Invalid neuron selection type: {self.neuron_select_type}")
        return synch_representation_size



    
    def forward(self, x, indexs, relations, device, threshold, lazy=False, track=False):
        # B = x.size(0)
        B = 1
        device = device
        patch_num = 0
        patch_num1 = 0
        patch_num2 = 0
        patch_num3 = 0
        patch_num4 = 0
        patch_num_1024 = 0
        pre_activations_tracking = []
        post_activations_tracking = []
        synch_out_tracking = []
        synch_action_tracking = []
        attention_tracking = []
        synchronisation_out_list = []

        # --- Featurise Input Data ---
        if lazy:
            kv_8192 = self.compute_features(x['8192'].unsqueeze(0).to(device))
            kv_4096 = self.compute_features(x['4096'].unsqueeze(0).to(device))
            kv_2048 = self.compute_features(x['2048'].unsqueeze(0).to(device))
            kv_1024 = self.compute_features(x['1024'].unsqueeze(0).to(device))
        else: 
            kv_8192 = self.compute_features(x['8192'].to(device))
            kv_4096 = self.compute_features(x['4096'].to(device))
            kv_2048 = self.compute_features(x['2048'].to(device))
            kv_1024 = self.compute_features(x['1024'].to(device))
            index_8192 = indexs['8192'].squeeze(0)
            index_4096 = indexs['4096'].squeeze(0)
            index_2048 = indexs['2048'].squeeze(0)
            # index_1024 = indexs['1024'].squeeze(0)
            relation_8192_4096 = relations['8192_4096']
            relation_4096_2048 = relations['4096_2048']
            relation_2048_1024 = relations['2048_1024']
            
        patch_num_1024 = kv_1024.shape[1]

        # --- Initialise Recurrent State ---
        state_trace = self.start_trace.unsqueeze(0).expand(B, -1, -1) 
        activated_state = self.start_activated_state.unsqueeze(0).expand(B, -1) 

        # --- Prepare Storage for Outputs per Iteration ---
        predictions = torch.empty(B, self.out_dims, self.iterations, device=device, dtype=torch.float32) 
        certainties = torch.empty(B, 2, self.iterations, device=device, dtype=torch.float32)   
        decay_alpha_action, decay_beta_action = None, None
        self.decay_params_action.data = torch.clamp(self.decay_params_action, 0, 15)  
        self.decay_params_out.data = torch.clamp(self.decay_params_out, 0, 15)
        r_action, r_out = torch.exp(-self.decay_params_action).unsqueeze(0).repeat(B, 1), torch.exp(-self.decay_params_out).unsqueeze(0).repeat(B, 1) 

       
        _, decay_alpha_out, decay_beta_out = self.compute_synchronisation(activated_state, None, None, r_out, synch_type='out')
        
        
        
  
        active_indices_4096 = None
        active_indices_2048 = None
        active_indices_1024 = None
        
        
        best_certainty_value = -1.0
        best_stepi = -1
        best_prediction = None
        best_certainty = None
        for stepi in range(self.iterations):   
            
            if stepi == 0:
                kv = kv_8192
                patch_num1 = kv.shape[1]
                
            elif stepi == 20:   
                index_1 = torch.argmax(certainties[..., 1, :20], dim=-1)  # shape: [B]
                att_weight = attention_tracking[index_1]

 
                if isinstance(att_weight, np.ndarray):
                    att_weight_tensor = torch.from_numpy(att_weight).to(device)
                else:
                    att_weight_tensor = att_weight.detach()

                
                avg_att_weight = torch.mean(att_weight_tensor, dim=1)  

                
                k = min(10, att_weight_tensor.size(-1))

                
                topk_values, topk_indices = torch.topk(avg_att_weight, k=k, dim=-1)
                
                
                topk_indices = topk_indices.view(-1)
 
                coords = index_8192[topk_indices.view(-1).cpu()]  

                next_selected_indices = []
                for coord in coords:
                    x, y = coord
                    coord_str = f'{x}_{y}_8192.png'
                    if coord_str in relation_8192_4096:
                        indices = relation_8192_4096[coord_str].squeeze(0).tolist()
                        next_selected_indices.extend(indices)
                

                active_indices_4096 = torch.unique(torch.tensor(next_selected_indices, device=device))
                active_indices_4096 = active_indices_4096.to(torch.long)
                kv = kv_4096[:, active_indices_4096, :]

                patch_num2 = kv.shape[1]

            elif stepi == 40:
                index_2 = torch.argmax(certainties[..., 1, 20:40], dim=-1)  
                att_weight = attention_tracking[index_2+20]


                if isinstance(att_weight, np.ndarray):
                    att_weight_tensor = torch.from_numpy(att_weight).to(device)
                else:
                    att_weight_tensor = att_weight.detach()


                avg_att_weight = torch.mean(att_weight_tensor, dim=1)  


                k = min(10, att_weight_tensor.size(-1))

     
                topk_values, topk_indices = torch.topk(avg_att_weight, k=k, dim=-1)
                
                
                topk_indices = topk_indices.view(-1)

                global_indices = active_indices_4096[topk_indices]
                coords = index_4096[global_indices.cpu()]


                next_selected_indices = []
                for coord in coords:
                    x, y = coord
                    coord_str = f'{x}_{y}_4096.png'
                    if coord_str in relation_4096_2048:
                        indices = relation_4096_2048[coord_str].squeeze(0).tolist()
                        next_selected_indices.extend(indices)
                

                active_indices_2048 = torch.unique(torch.tensor(next_selected_indices, device=device))
                active_indices_2048 = active_indices_2048.to(torch.long)
                kv = kv_2048[:, active_indices_2048, :]
   
                patch_num3 = kv.shape[1]
                
            elif stepi == 60:
                index_3 = torch.argmax(certainties[..., 1, 40:60], dim=-1)
                att_weight = attention_tracking[index_3 + 40]

                if isinstance(att_weight, np.ndarray):
                    att_weight_tensor = torch.from_numpy(att_weight).to(device)
                else:
                    att_weight_tensor = att_weight.detach()

                avg_att_weight = torch.mean(att_weight_tensor, dim=1)
                
                
                k = min(10, att_weight_tensor.size(-1))
                _, topk_indices = torch.topk(avg_att_weight, k=k, dim=-1)
                topk_indices = topk_indices.view(-1)

                global_indices = active_indices_2048[topk_indices]
                coords = index_2048[global_indices.cpu()]


                next_selected_indices = []
                for coord in coords:
                    x, y = coord
                    coord_str = f'{x}_{y}_2048.png'
                    if coord_str in relation_2048_1024:
                        indices = relation_2048_1024[coord_str].squeeze(0).tolist()
                        next_selected_indices.extend(indices)
                
               
                active_indices_1024 = torch.unique(torch.tensor(next_selected_indices, device=device))
                active_indices_1024 = active_indices_1024.to(torch.long)
                kv = kv_1024[:, active_indices_1024, :]

                patch_num4 = kv.shape[1]


            
            synchronisation_action, decay_alpha_action, decay_beta_action = self.compute_synchronisation(activated_state, decay_alpha_action, decay_beta_action, r_action, synch_type='action')

            # --- Interact with Data via Attention ---
            q = self.q_proj(synchronisation_action).unsqueeze(1) 

            pre_synapse_input = torch.concatenate((attn_out, activated_state), dim=-1) 
            
            # --- Apply Synapses ---
            state = self.synapses(pre_synapse_input)  
            # The 'state_trace' is the history of incoming pre-activations
            state_trace = torch.cat((state_trace[:, :, 1:], state.unsqueeze(-1)), dim=-1) # 

            # --- Apply Neuron-Level Models ---
            activated_state = self.trace_processor(state_trace)    #
            synchronisation_out, decay_alpha_out, decay_beta_out = self.compute_synchronisation(activated_state, decay_alpha_out, decay_beta_out, r_out, synch_type='out')
            
            
            ################# idea 3 ################
            if stepi >= 60:
                synchronisation_out_2048 = synchronisation_out_list[index_3+40]
                synchronisation_out = synchronisation_out + synchronisation_out_2048
            elif stepi >= 40:
                synchronisation_out_4096 = synchronisation_out_list[index_2+20]
                synchronisation_out = synchronisation_out + synchronisation_out_4096
            elif stepi >= 20:
                synchronisation_out_8192 = synchronisation_out_list[index_1]
                synchronisation_out = synchronisation_out + synchronisation_out_8192
            else: 
                synchronisation_out = synchronisation_out
            
            #########################################
            
            # --- Get Predictions and Certainties ---

            current_prediction = self.output_projector(synchronisation_out)
            current_certainty = self.compute_certainty(current_prediction)

            predictions[..., stepi] = current_prediction
            certainties[..., stepi] = current_certainty
            
            #######################
            attention_tracking.append(attn_weights.detach().cpu().numpy())
            synchronisation_out_list.append(synchronisation_out)
            ######################
            

            mean_certainty = current_certainty[:, 1].mean()  
            # threshold = 0.5
            if mean_certainty >= threshold:
                patch_num = patch_num1 + patch_num2 + patch_num3 + patch_num4
                return current_prediction, current_certainty, stepi, patch_num, patch_num_1024
            

            if mean_certainty > best_certainty_value:
                best_certainty_value = mean_certainty
                best_stepi = stepi
                best_prediction = current_prediction
                best_certainty = current_certainty
        
        patch_num = patch_num1 + patch_num2 + patch_num3 + patch_num4
        
        return best_prediction, best_certainty, best_stepi, patch_num, patch_num_1024

    ##########################################################################################################################################