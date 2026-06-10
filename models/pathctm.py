import math

import numpy as np
import torch
import torch.nn as nn

from models.constants import (
    VALID_BACKBONE_TYPES,
    VALID_NEURON_SELECT_TYPES,
    VALID_POSITIONAL_EMBEDDING_TYPES,
)
from models.modules import (
    CustomRotationalEmbedding,
    CustomRotationalEmbedding1D,
    LearnableFourierPositionalEncoding,
    MultiLearnableFourierPositionalEncoding,
    ParityBackbone,
    ShallowWide,
    SuperLinear,
    Squeeze,
    SynapseUNET,
)
from models.resnet import prepare_resnet_backbone
from models.utils import compute_normalized_entropy


class PatchCTM(nn.Module):
    def __init__(
        self,
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
        prediction_reshaper=None,
        dropout=0,
        dropout_nlm=None,
        neuron_select_type="random-pairing",
        n_random_pairing_self=0,
    ):
        super().__init__()

        self.iterations = iterations
        self.d_model = d_model
        self.d_input = d_input
        self.memory_length = memory_length
        self.prediction_reshaper = prediction_reshaper or [-1]
        self.n_synch_out = n_synch_out
        self.n_synch_action = n_synch_action
        self.backbone_type = backbone_type
        self.out_dims = out_dims
        self.positional_embedding_type = positional_embedding_type
        self.neuron_select_type = neuron_select_type
        dropout_nlm = dropout if dropout_nlm is None else dropout_nlm

        self.verify_args()

        d_backbone = self.get_d_backbone()
        self.set_initial_rgb()
        self.set_backbone()
        self.positional_embedding = self.get_positional_embedding(d_backbone)
        self.kv_proj = nn.Sequential(nn.LazyLinear(self.d_input), nn.LayerNorm(self.d_input)) if heads else None
        self.q_proj = nn.LazyLinear(self.d_input) if heads else None
        self.attention = nn.MultiheadAttention(self.d_input, heads, dropout, batch_first=True) if heads else None
        self.synapses = self.get_synapses(synapse_depth, d_model, dropout)
        self.trace_processor = self.get_neuron_level_models(
            deep_nlms,
            do_layernorm_nlm,
            memory_length,
            memory_hidden_dims,
            d_model,
            dropout_nlm,
        )

        self.register_parameter(
            "start_activated_state",
            nn.Parameter(torch.zeros(d_model).uniform_(-math.sqrt(1 / d_model), math.sqrt(1 / d_model))),
        )
        self.register_parameter(
            "start_trace",
            nn.Parameter(
                torch.zeros(d_model, memory_length).uniform_(
                    -math.sqrt(1 / (d_model + memory_length)),
                    math.sqrt(1 / (d_model + memory_length)),
                )
            ),
        )

        self.synch_representation_size_action = self.calculate_synch_representation_size(self.n_synch_action)
        self.synch_representation_size_out = self.calculate_synch_representation_size(self.n_synch_out)

        if self.synch_representation_size_action:
            self.set_synchronisation_parameters("action", self.n_synch_action, n_random_pairing_self)
        self.set_synchronisation_parameters("out", self.n_synch_out, n_random_pairing_self)

        self.output_projector = nn.Sequential(nn.LazyLinear(self.out_dims))

    def compute_synchronisation(self, activated_state, decay_alpha, decay_beta, r, synch_type):
        if synch_type == "action":
            n_synch = self.n_synch_action
            neuron_indices_left = self.action_neuron_indices_left
            neuron_indices_right = self.action_neuron_indices_right
        elif synch_type == "out":
            n_synch = self.n_synch_out
            neuron_indices_left = self.out_neuron_indices_left
            neuron_indices_right = self.out_neuron_indices_right
        else:
            raise ValueError(f"Invalid synch_type: {synch_type}")

        if self.neuron_select_type in ("first-last", "random"):
            if self.neuron_select_type == "first-last":
                if synch_type == "action":
                    selected_left = selected_right = activated_state[:, -n_synch:]
                else:
                    selected_left = selected_right = activated_state[:, :n_synch]
            else:
                selected_left = activated_state[:, neuron_indices_left]
                selected_right = activated_state[:, neuron_indices_right]

            outer = selected_left.unsqueeze(2) * selected_right.unsqueeze(1)
            i, j = torch.triu_indices(n_synch, n_synch)
            pairwise_product = outer[:, i, j]
        elif self.neuron_select_type == "random-pairing":
            left = activated_state[:, neuron_indices_left]
            right = activated_state[:, neuron_indices_right]
            pairwise_product = left * right
        else:
            raise ValueError(f"Invalid neuron selection type: {self.neuron_select_type}")

        if decay_alpha is None or decay_beta is None:
            decay_alpha = pairwise_product
            decay_beta = torch.ones_like(pairwise_product)
        else:
            decay_alpha = r * decay_alpha + pairwise_product
            decay_beta = r * decay_beta + 1

        synchronisation = decay_alpha / torch.sqrt(decay_beta)
        return synchronisation, decay_alpha, decay_beta

    def compute_features(self, x):
        self.kv_features = x
        return self.kv_proj(self.kv_features)

    def compute_certainty(self, current_prediction):
        batch_size = current_prediction.size(0)
        reshaped_pred = current_prediction.reshape([batch_size] + self.prediction_reshaper)
        ne = compute_normalized_entropy(reshaped_pred)
        return torch.stack((ne, 1 - ne), -1)

    def set_initial_rgb(self):
        if "resnet" in self.backbone_type:
            self.initial_rgb = nn.LazyConv2d(3, 1, 1)
        else:
            self.initial_rgb = nn.Identity()

    def get_d_backbone(self):
        if self.backbone_type == "shallow-wide":
            return 2048
        if self.backbone_type == "parity_backbone":
            return self.d_input
        if "resnet" in self.backbone_type:
            stage = self.backbone_type.split("-")[1]
            if "18" in self.backbone_type or "34" in self.backbone_type:
                stage_map = {"1": 64, "2": 128, "3": 256, "4": 512}
            else:
                stage_map = {"1": 256, "2": 512, "3": 1024, "4": 2048}
            if stage not in stage_map:
                raise NotImplementedError
            return stage_map[stage]
        if self.backbone_type == "none":
            return None
        raise ValueError(f"Invalid backbone_type: {self.backbone_type}")

    def set_backbone(self):
        if self.backbone_type == "shallow-wide":
            self.backbone = ShallowWide()
        elif self.backbone_type == "parity_backbone":
            self.backbone = ParityBackbone(n_embeddings=2, d_embedding=self.get_d_backbone())
        elif "resnet" in self.backbone_type:
            self.backbone = prepare_resnet_backbone(self.backbone_type)
        elif self.backbone_type == "none":
            self.backbone = nn.Identity()
        else:
            raise ValueError(f"Invalid backbone_type: {self.backbone_type}")

    def get_positional_embedding(self, d_backbone):
        if self.positional_embedding_type == "learnable-fourier":
            return LearnableFourierPositionalEncoding(d_backbone, gamma=1 / 2.5)
        if self.positional_embedding_type == "multi-learnable-fourier":
            return MultiLearnableFourierPositionalEncoding(d_backbone)
        if self.positional_embedding_type == "custom-rotational":
            return CustomRotationalEmbedding(d_backbone)
        if self.positional_embedding_type == "custom-rotational-1d":
            return CustomRotationalEmbedding1D(d_backbone)
        if self.positional_embedding_type == "none":
            return lambda x: 0
        raise ValueError(f"Invalid positional_embedding_type: {self.positional_embedding_type}")

    def get_neuron_level_models(
        self,
        deep_nlms,
        do_layernorm_nlm,
        memory_length,
        memory_hidden_dims,
        d_model,
        dropout,
    ):
        if deep_nlms:
            return nn.Sequential(
                nn.Sequential(
                    SuperLinear(
                        in_dims=memory_length,
                        out_dims=2 * memory_hidden_dims,
                        N=d_model,
                        do_norm=do_layernorm_nlm,
                        dropout=dropout,
                    ),
                    nn.GLU(),
                    SuperLinear(
                        in_dims=memory_hidden_dims,
                        out_dims=2,
                        N=d_model,
                        do_norm=do_layernorm_nlm,
                        dropout=dropout,
                    ),
                    nn.GLU(),
                    Squeeze(-1),
                )
            )

        return nn.Sequential(
            nn.Sequential(
                SuperLinear(
                    in_dims=memory_length,
                    out_dims=2,
                    N=d_model,
                    do_norm=do_layernorm_nlm,
                    dropout=dropout,
                ),
                nn.GLU(),
                Squeeze(-1),
            )
        )

    def get_synapses(self, synapse_depth, d_model, dropout):
        if synapse_depth == 1:
            return nn.Sequential(
                nn.Dropout(dropout),
                nn.LazyLinear(d_model * 2),
                nn.GLU(),
                nn.LayerNorm(d_model),
            )
        return SynapseUNET(d_model, synapse_depth, 16, dropout)

    def set_synchronisation_parameters(self, synch_type, n_synch, n_random_pairing_self=0):
        if synch_type not in ("out", "action"):
            raise ValueError(f"Invalid synch_type: {synch_type}")
        left, right = self.initialize_left_right_neurons(
            synch_type,
            self.d_model,
            n_synch,
            n_random_pairing_self,
        )
        if synch_type == "action":
            synch_representation_size = self.synch_representation_size_action
        else:
            synch_representation_size = self.synch_representation_size_out
        self.register_buffer(f"{synch_type}_neuron_indices_left", left)
        self.register_buffer(f"{synch_type}_neuron_indices_right", right)
        self.register_parameter(
            f"decay_params_{synch_type}",
            nn.Parameter(torch.zeros(synch_representation_size), requires_grad=True),
        )

    def initialize_left_right_neurons(self, synch_type, d_model, n_synch, n_random_pairing_self=0):
        if self.neuron_select_type == "first-last":
            if synch_type == "out":
                neuron_indices_left = neuron_indices_right = torch.arange(0, n_synch)
            else:
                neuron_indices_left = neuron_indices_right = torch.arange(d_model - n_synch, d_model)
        elif self.neuron_select_type == "random":
            neuron_indices_left = torch.from_numpy(np.random.choice(np.arange(d_model), size=n_synch))
            neuron_indices_right = torch.from_numpy(np.random.choice(np.arange(d_model), size=n_synch))
        elif self.neuron_select_type == "random-pairing":
            if n_synch <= n_random_pairing_self:
                raise AssertionError(
                    f"Need at least {n_random_pairing_self} pairs for {self.neuron_select_type}"
                )
            neuron_indices_left = torch.from_numpy(np.random.choice(np.arange(d_model), size=n_synch))
            neuron_indices_right = torch.concatenate(
                (
                    neuron_indices_left[:n_random_pairing_self],
                    torch.from_numpy(np.random.choice(np.arange(d_model), size=n_synch - n_random_pairing_self)),
                )
            )
        else:
            raise ValueError(f"Invalid neuron selection type: {self.neuron_select_type}")

        device = self.start_activated_state.device
        return neuron_indices_left.to(device), neuron_indices_right.to(device)

    def verify_args(self):
        if self.neuron_select_type not in VALID_NEURON_SELECT_TYPES:
            raise AssertionError(f"Invalid neuron selection type: {self.neuron_select_type}")
        if self.backbone_type not in VALID_BACKBONE_TYPES + ["none"]:
            raise AssertionError(f"Invalid backbone_type: {self.backbone_type}")
        if self.positional_embedding_type not in VALID_POSITIONAL_EMBEDDING_TYPES + ["none"]:
            raise AssertionError(f"Invalid positional_embedding_type: {self.positional_embedding_type}")
        if self.neuron_select_type == "first-last" and self.d_model < (self.n_synch_out + self.n_synch_action):
            raise AssertionError("d_model must be >= n_synch_out + n_synch_action for neuron subsets")
        if self.backbone_type == "none" and self.positional_embedding_type != "none":
            raise AssertionError("There should be no positional embedding if there is no backbone.")

    def calculate_synch_representation_size(self, n_synch):
        if self.neuron_select_type == "random-pairing":
            return n_synch
        if self.neuron_select_type in ("first-last", "random"):
            return (n_synch * (n_synch + 1)) // 2
        raise ValueError(f"Invalid neuron selection type: {self.neuron_select_type}")

    def _prepare_kv(self, x, device, lazy):
        if lazy:
            return (
                self.compute_features(x["8192"].unsqueeze(0).to(device)),
                self.compute_features(x["4096"].unsqueeze(0).to(device)),
                self.compute_features(x["2048"].unsqueeze(0).to(device)),
                self.compute_features(x["1024"].unsqueeze(0).to(device)),
            )
        return (
            self.compute_features(x["8192"].to(device)),
            self.compute_features(x["4096"].to(device)),
            self.compute_features(x["2048"].to(device)),
            self.compute_features(x["1024"].to(device)),
        )

    @staticmethod
    def _to_attention_tensor(attn_weights, device):
        if isinstance(attn_weights, np.ndarray):
            return torch.from_numpy(attn_weights).to(device)
        return attn_weights.detach()

    def _select_topk_indices(self, attn_weights, topk, device):
        attn_tensor = self._to_attention_tensor(attn_weights, device)
        avg_att_weight = torch.mean(attn_tensor, dim=1)
        k = min(topk, attn_tensor.size(-1))
        return torch.topk(avg_att_weight, k=k, dim=-1).indices.view(-1)

    @staticmethod
    def _expand_relation_indices(coords, relation_map, suffix_candidates, device):
        next_selected_indices = []
        for coord in coords:
            x_coord = int(coord[0].item())
            y_coord = int(coord[1].item())
            coord_str = None
            for suffix in suffix_candidates:
                candidate = f"{x_coord}_{y_coord}_{suffix}.png"
                if candidate in relation_map:
                    coord_str = candidate
                    break
            if coord_str is None:
                coord_prefix = f"{x_coord}_{y_coord}_"
                coord_str = next((key for key in relation_map if key.startswith(coord_prefix)), None)
            if coord_str is not None:
                next_selected_indices.extend(relation_map[coord_str].reshape(-1).tolist())
        if not next_selected_indices:
            return torch.empty(0, dtype=torch.long, device=device)
        return torch.unique(torch.tensor(next_selected_indices, dtype=torch.long, device=device))

    def forward(self, x, indexs, relations, device, lazy=False, track=False):
        batch_size = 1
        topk = 10

        tracked_pre_activations = []
        tracked_post_activations = []
        tracked_synch_out = []
        tracked_synch_action = []
        tracked_attention = []
        attention_history = []
        synchronisation_out_history = []

        kv_8192, kv_4096, kv_2048, kv_1024 = self._prepare_kv(x, device, lazy)
        if not lazy:
            index_8192 = indexs["8192"].squeeze(0)
            index_4096 = indexs["4096"].squeeze(0)
            index_2048 = indexs["2048"].squeeze(0)
            relation_8192_4096 = relations["8192_4096"]
            relation_4096_2048 = relations["4096_2048"]
            relation_2048_1024 = relations["2048_1024"]

        state_trace = self.start_trace.unsqueeze(0).expand(batch_size, -1, -1)
        activated_state = self.start_activated_state.unsqueeze(0).expand(batch_size, -1)

        predictions = torch.empty(batch_size, self.out_dims, self.iterations, device=device, dtype=torch.float32)
        certainties = torch.empty(batch_size, 2, self.iterations, device=device, dtype=torch.float32)

        decay_alpha_action, decay_beta_action = None, None
        self.decay_params_action.data = torch.clamp(self.decay_params_action, 0, 15)
        self.decay_params_out.data = torch.clamp(self.decay_params_out, 0, 15)
        r_action = torch.exp(-self.decay_params_action).unsqueeze(0).repeat(batch_size, 1)
        r_out = torch.exp(-self.decay_params_out).unsqueeze(0).repeat(batch_size, 1)

        _, decay_alpha_out, decay_beta_out = self.compute_synchronisation(
            activated_state,
            None,
            None,
            r_out,
            synch_type="out",
        )

        kv = kv_8192
        active_indices_4096 = None
        active_indices_2048 = None
        index_1 = 0
        index_2 = 0
        index_3 = 0

        for stepi in range(self.iterations):
            if stepi == 20:
                index_1 = int(torch.argmax(certainties[..., 1, :20], dim=-1).item())
                topk_indices = self._select_topk_indices(attention_history[index_1], topk, device)
                coords = index_8192[topk_indices.cpu()]
                active_indices_4096 = self._expand_relation_indices(
                    coords,
                    relation_8192_4096,
                    ("8192",),
                    device,
                )
                kv = kv_4096[:, active_indices_4096, :]
            elif stepi == 40:
                index_2 = int(torch.argmax(certainties[..., 1, 20:40], dim=-1).item())
                topk_indices = self._select_topk_indices(attention_history[index_2 + 20], topk, device)
                global_indices = active_indices_4096[topk_indices]
                coords = index_4096[global_indices.cpu().long()]
                active_indices_2048 = self._expand_relation_indices(
                    coords,
                    relation_4096_2048,
                    ("4096", "1024"),
                    device,
                )
                kv = kv_2048[:, active_indices_2048, :]
            elif stepi == 60:
                index_3 = int(torch.argmax(certainties[..., 1, 40:60], dim=-1).item())
                topk_indices = self._select_topk_indices(attention_history[index_3 + 40], topk, device)
                global_indices = active_indices_2048[topk_indices]
                coords = index_2048[global_indices.cpu().long()]
                selected_indices_1024 = self._expand_relation_indices(
                    coords,
                    relation_2048_1024,
                    ("2048", "512"),
                    device,
                )
                kv = kv_1024[:, selected_indices_1024, :]

            synchronisation_action, decay_alpha_action, decay_beta_action = self.compute_synchronisation(
                activated_state,
                decay_alpha_action,
                decay_beta_action,
                r_action,
                synch_type="action",
            )

            q = self.q_proj(synchronisation_action).unsqueeze(1)
            attn_out, attn_weights = self.attention(q, kv, kv, average_attn_weights=False, need_weights=True)
            attn_out = attn_out.squeeze(1)
            pre_synapse_input = torch.concatenate((attn_out, activated_state), dim=-1)
            state = self.synapses(pre_synapse_input)
            state_trace = torch.cat((state_trace[:, :, 1:], state.unsqueeze(-1)), dim=-1)
            activated_state = self.trace_processor(state_trace)

            synchronisation_out, decay_alpha_out, decay_beta_out = self.compute_synchronisation(
                activated_state,
                decay_alpha_out,
                decay_beta_out,
                r_out,
                synch_type="out",
            )

            if stepi >= 60:
                synchronisation_out = synchronisation_out + synchronisation_out_history[index_3 + 40]
            elif stepi >= 40:
                synchronisation_out = synchronisation_out + synchronisation_out_history[index_2 + 20]
            elif stepi >= 20:
                synchronisation_out = synchronisation_out + synchronisation_out_history[index_1]

            current_prediction = self.output_projector(synchronisation_out)
            current_certainty = self.compute_certainty(current_prediction)

            predictions[..., stepi] = current_prediction
            certainties[..., stepi] = current_certainty

            attention_snapshot = attn_weights.detach().cpu().numpy()
            attention_history.append(attention_snapshot)
            synchronisation_out_history.append(synchronisation_out)

            if track:
                tracked_pre_activations.append(state_trace[:, :, -1].detach().cpu().numpy())
                tracked_post_activations.append(activated_state.detach().cpu().numpy())
                tracked_attention.append(attention_snapshot)
                tracked_synch_out.append(synchronisation_out.detach().cpu().numpy())
                tracked_synch_action.append(synchronisation_action.detach().cpu().numpy())

        if track:
            return (
                predictions,
                certainties,
                (np.array(tracked_synch_out), np.array(tracked_synch_action)),
                np.array(tracked_pre_activations),
                np.array(tracked_post_activations),
                np.array(tracked_attention),
            )
        return predictions, certainties, synchronisation_out
