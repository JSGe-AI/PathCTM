import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from models.utils import add_coord_dim

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Squeeze(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(self.dim)

class SynapseUNET(nn.Module):
    def __init__(self,
                 out_dims,
                 depth,
                 minimum_width=16,
                 dropout=0.0):
        super().__init__()
        self.width_out = out_dims
        self.n_deep = depth

        widths = np.linspace(out_dims, minimum_width, depth)

        self.first_projection = nn.Sequential(
            nn.LazyLinear(int(widths[0])),
            nn.LayerNorm(int(widths[0])),
            nn.SiLU()
        )

        self.down_projections = nn.ModuleList()
        self.up_projections = nn.ModuleList()
        self.skip_lns = nn.ModuleList()
        num_blocks = len(widths) - 1

        for i in range(num_blocks):
            self.down_projections.append(nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(int(widths[i]), int(widths[i+1])),
                nn.LayerNorm(int(widths[i+1])),
                nn.SiLU()
            ))
            self.up_projections.append(nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(int(widths[i+1]), int(widths[i])),
                nn.LayerNorm(int(widths[i])),
                nn.SiLU()
            ))
            self.skip_lns.append(nn.LayerNorm(int(widths[i])))

    def forward(self, x):
        out_first = self.first_projection(x)

        outs_down = [out_first]
        for layer in self.down_projections:
            outs_down.append(layer(outs_down[-1]))

        outs_up = outs_down[-1]
        num_blocks = len(self.up_projections)

        for i in range(num_blocks):
            up_layer_idx = num_blocks - 1 - i
            out_up = self.up_projections[up_layer_idx](outs_up)

            skip_idx = up_layer_idx
            skip_connection = outs_down[skip_idx]

            outs_up = self.skip_lns[skip_idx](out_up + skip_connection)

        return outs_up


class SuperLinear(nn.Module):
    def __init__(self,
                 in_dims,
                 out_dims,
                 N,
                 T=1.0,
                 do_norm=False,
                 dropout=0):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else Identity()
        self.in_dims = in_dims
        self.layernorm = nn.LayerNorm(in_dims, elementwise_affine=True) if do_norm else Identity()
        self.do_norm = do_norm

        self.register_parameter('w1', nn.Parameter(
            torch.empty((in_dims, out_dims, N)).uniform_(
                -1/math.sqrt(in_dims + out_dims),
                 1/math.sqrt(in_dims + out_dims)
            ), requires_grad=True)
        )
        self.register_parameter('b1', nn.Parameter(torch.zeros((1, N, out_dims)), requires_grad=True))
        self.register_parameter('T', nn.Parameter(torch.Tensor([T]))) 

    def forward(self, x):
        out = self.dropout(x)
        out = self.layernorm(out)

        out = torch.einsum('BDM,MHD->BDH', out, self.w1) + self.b1

        out = out.squeeze(-1) / self.T
        return out


class ParityBackbone(nn.Module):
    def __init__(self, n_embeddings, d_embedding):
        super(ParityBackbone, self).__init__()
        self.embedding = nn.Embedding(n_embeddings, d_embedding)

    def forward(self, x):
        x = (x == 1).long() 
        return self.embedding(x.long()).transpose(1, 2)

class QAMNISTOperatorEmbeddings(nn.Module):
    def __init__(self, num_operator_types, d_projection):
        super(QAMNISTOperatorEmbeddings, self).__init__()
        self.embedding = nn.Embedding(num_operator_types, d_projection)

    def forward(self, x):
        return self.embedding(-x - 1)

class QAMNISTIndexEmbeddings(torch.nn.Module):
    def __init__(self, max_seq_length, embedding_dim):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.embedding_dim = embedding_dim

        embedding = torch.zeros(max_seq_length, embedding_dim)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        
        embedding[:, 0::2] = torch.sin(position * div_term)
        embedding[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('embedding', embedding)

    def forward(self, x):
        return self.embedding[x]
    
class ThoughtSteps:

    def __init__(self, iterations_per_digit, iterations_per_question_part, total_iterations_for_answering, total_iterations_for_digits, total_iterations_for_question):
        self.iterations_per_digit = iterations_per_digit
        self.iterations_per_question_part = iterations_per_question_part
        self.total_iterations_for_digits = total_iterations_for_digits
        self.total_iterations_for_question = total_iterations_for_question
        self.total_iterations_for_answering = total_iterations_for_answering
        self.total_iterations = self.total_iterations_for_digits + self.total_iterations_for_question + self.total_iterations_for_answering

    def determine_step_type(self, stepi: int):
        is_digit_step = stepi < self.total_iterations_for_digits
        is_question_step = self.total_iterations_for_digits <= stepi < self.total_iterations_for_digits + self.total_iterations_for_question
        is_answer_step = stepi >= self.total_iterations_for_digits + self.total_iterations_for_question
        return is_digit_step, is_question_step, is_answer_step

    def determine_answer_step_type(self, stepi: int):
        step_within_questions = stepi - self.total_iterations_for_digits
        if step_within_questions % (2 * self.iterations_per_question_part) < self.iterations_per_question_part:
            is_index_step = True
            is_operator_step = False
        else:
            is_index_step = False
            is_operator_step = True
        return is_index_step, is_operator_step

class MNISTBackbone(nn.Module):
    def __init__(self, d_input):
        super(MNISTBackbone, self).__init__()
        self.layers = nn.Sequential(
            nn.LazyConv2d(d_input, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(d_input),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.LazyConv2d(d_input, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(d_input),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

    def forward(self, x):
        return self.layers(x)


class MiniGridBackbone(nn.Module):
    def __init__(self, d_input, grid_size=7, num_objects=11, num_colors=6, num_states=3, embedding_dim=8):
        super().__init__()
        self.object_embedding = nn.Embedding(num_objects, embedding_dim)
        self.color_embedding = nn.Embedding(num_colors, embedding_dim)
        self.state_embedding = nn.Embedding(num_states, embedding_dim)
        
        self.position_embedding = nn.Embedding(grid_size * grid_size, embedding_dim)

        self.project_to_d_projection = nn.Sequential(
            nn.Linear(embedding_dim * 4, d_input * 2),
            nn.GLU(),
            nn.LayerNorm(d_input),
            nn.Linear(d_input, d_input * 2),
            nn.GLU(),
            nn.LayerNorm(d_input)
        )

    def forward(self, x):
        x = x.long()
        B, H, W, C = x.size()

        object_idx = x[:,:,:, 0]
        color_idx =  x[:,:,:, 1]
        state_idx =  x[:,:,:, 2]

        obj_embed = self.object_embedding(object_idx)
        color_embed = self.color_embedding(color_idx)
        state_embed = self.state_embedding(state_idx)
        
        pos_idx = torch.arange(H * W, device=x.device).view(1, H, W).expand(B, -1, -1)
        pos_embed = self.position_embedding(pos_idx)

        out = self.project_to_d_projection(torch.cat([obj_embed, color_embed, state_embed, pos_embed], dim=-1))
        return out

class ClassicControlBackbone(nn.Module):
    def __init__(self, d_input):
        super().__init__()
        self.input_projector = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(d_input * 2),
            nn.GLU(),
            nn.LayerNorm(d_input),
            nn.LazyLinear(d_input * 2),
            nn.GLU(),
            nn.LayerNorm(d_input)
        )

    def forward(self, x):
        return self.input_projector(x)


class ShallowWide(nn.Module):
    def __init__(self):
        super(ShallowWide, self).__init__()
        self.layers = nn.Sequential(
            nn.LazyConv2d(4096, kernel_size=3, stride=2, padding=1),
            nn.GLU(dim=1),
            nn.BatchNorm2d(2048),
            nn.Conv2d(2048, 4096, kernel_size=3, stride=1, padding=1, groups=32),
            nn.GLU(dim=1),
            nn.BatchNorm2d(2048)
        )
    def forward(self, x):
        return self.layers(x)


class PretrainedResNetWrapper(nn.Module):
    def __init__(self, resnet_type, fine_tune=True):
        super(PretrainedResNetWrapper, self).__init__()
        self.resnet_type = resnet_type
        self.backbone = torch.hub.load('', resnet_type, pretrained=True)

        if not fine_tune:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.backbone.avgpool = Identity()
        self.backbone.fc = Identity()

    def forward(self, x):
        out = self.backbone(x)

        nc = 256 if ('18' in self.resnet_type or '34' in self.resnet_type) else 512 if '50' in self.resnet_type else 1024 if '101' in self.resnet_type else 2048
        num_features = out.shape[-1]
        wh_squared = num_features / nc
        if wh_squared < 0 or not float(wh_squared).is_integer():
             print(f"nc={nc}, num_features={num_features}")
             return out
        wh = int(np.sqrt(wh_squared))

        return out.reshape(x.size(0), nc, wh, wh)

class LearnableFourierPositionalEncoding(nn.Module):
    def __init__(self, d_model,
                 G=1, M=2,
                 F_dim=256,
                 H_dim=128,
                 gamma=1/2.5,
                 ):
        super().__init__()
        self.G = G
        self.M = M
        self.F_dim = F_dim
        self.H_dim = H_dim
        self.D = d_model
        self.gamma = gamma

        self.Wr = nn.Linear(self.M, self.F_dim // 2, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(self.F_dim, self.H_dim, bias=True),
            nn.GLU(),
            nn.Linear(self.H_dim // 2, self.D // self.G),
            nn.LayerNorm(self.D // self.G)
        )

        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.Wr.weight.data, mean=0, std=self.gamma ** -2)

    def forward(self, x):
        B, C, H, W = x.shape
        x_coord = add_coord_dim(x[:,0])

        projected = self.Wr(x_coord)
        cosines = torch.cos(projected)
        sines = torch.sin(projected)
        F = (1.0 / math.sqrt(self.F_dim)) * torch.cat([cosines, sines], dim=-1)

        Y = self.mlp(F)

        PEx = Y.permute(0, 3, 1, 2)
        return PEx


class MultiLearnableFourierPositionalEncoding(nn.Module):
    def __init__(self, d_model,
                 G=1, M=2,
                 F_dim=256,
                 H_dim=128,
                 gamma_range=[1.0, 0.1],
                 N=10,
                 ):
        super().__init__()
        self.embedders = nn.ModuleList()
        for gamma in np.linspace(gamma_range[0], gamma_range[1], N):
            self.embedders.append(LearnableFourierPositionalEncoding(d_model, G, M, F_dim, H_dim, gamma))

        self.register_parameter('combination', torch.nn.Parameter(torch.ones(N), requires_grad=True))
        self.N = N


    def forward(self, x):
        pos_embs = torch.stack([emb(x) for emb in self.embedders], dim=0)

        weights = F.softmax(self.combination, dim=-1).view(self.N, 1, 1, 1, 1)

        combined_emb = (pos_embs * weights).sum(0)
        return combined_emb


class CustomRotationalEmbedding(nn.Module):
    def __init__(self, d_model):
        super(CustomRotationalEmbedding, self).__init__()
        self.register_parameter('start_vector', nn.Parameter(torch.Tensor([0, 1]), requires_grad=True))
        self.projection = nn.Sequential(nn.Linear(4, d_model))

    def forward(self, x):
        B, C, H, W = x.shape
        device = x.device

        theta_rad = torch.deg2rad(torch.linspace(0, 180, W, device=device))
        cos_theta = torch.cos(theta_rad)
        sin_theta = torch.sin(theta_rad)

        rotation_matrices = torch.stack([
            torch.stack([cos_theta, -sin_theta], dim=-1),
            torch.stack([sin_theta, cos_theta], dim=-1)
        ], dim=1)

        rotated_vectors = torch.einsum('wij,j->wi', rotation_matrices, self.start_vector)

        key = torch.cat((
            torch.repeat_interleave(rotated_vectors.unsqueeze(1), W, dim=1),
            torch.repeat_interleave(rotated_vectors.unsqueeze(0), W, dim=0)
        ), dim=-1)

        pe_grid = self.projection(key)

        pe = pe_grid.permute(2, 0, 1).unsqueeze(0)

        if H != W:
            pass

        return pe

class CustomRotationalEmbedding1D(nn.Module):
    def __init__(self, d_model):
        super(CustomRotationalEmbedding1D, self).__init__()
        self.projection = nn.Linear(2, d_model)

    def forward(self, x):
        start_vector = torch.tensor([0., 1.], device=x.device, dtype=torch.float)
        theta_rad = torch.deg2rad(torch.linspace(0, 180, x.size(2), device=x.device))
        cos_theta = torch.cos(theta_rad)
        sin_theta = torch.sin(theta_rad)
        cos_theta = cos_theta.unsqueeze(1)
        sin_theta = sin_theta.unsqueeze(1)

        rotation_matrices = torch.stack([
        torch.cat([cos_theta, -sin_theta], dim=1),
        torch.cat([sin_theta, cos_theta], dim=1)
        ], dim=1)

        rotated_vectors = torch.einsum('bij,j->bi', rotation_matrices, start_vector)

        pe = self.projection(rotated_vectors)
        pe = torch.repeat_interleave(pe.unsqueeze(0), x.size(0), 0)
        return pe.transpose(1, 2)