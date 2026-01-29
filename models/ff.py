import torch.nn as nn


from models.modules import *
from models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152


class FFBaseline(nn.Module):


    def __init__(self,
                 d_model,
                 backbone_type,
                 out_dims,
                 dropout=0,
                 ):
        super(FFBaseline, self).__init__()

 
        self.d_model = d_model
        self.backbone_type = backbone_type
        self.out_dims = out_dims


        assert backbone_type in ['resnet18-1', 'resnet18-2', 'resnet18-3', 'resnet18-4',
                                 'resnet34-1', 'resnet34-2', 'resnet34-3', 'resnet34-4',
                                 'resnet50-1', 'resnet50-2', 'resnet50-3', 'resnet50-4',
                                 'resnet101-1', 'resnet101-2', 'resnet101-3', 'resnet101-4',
                                 'resnet152-1', 'resnet152-2', 'resnet152-3', 'resnet152-4',
                                 'none', 'shallow-wide', 'parity_backbone'], f"Invalid backbone_type: {backbone_type}"


        self.initial_rgb = Identity() 

        
        self.initial_rgb = nn.LazyConv2d(3, 1, 1) 
        resnet_family = resnet18 # Default
        if '34' in self.backbone_type: resnet_family = resnet34
        if '50' in self.backbone_type: resnet_family = resnet50
        if '101' in self.backbone_type: resnet_family = resnet101
        if '152' in self.backbone_type: resnet_family = resnet152

    
        block_num_str = self.backbone_type.split('-')[-1]
        hyper_blocks_to_keep = list(range(1, int(block_num_str) + 1)) if block_num_str.isdigit() else [1, 2, 3, 4]

        self.backbone = resnet_family(
            3, 
            hyper_blocks_to_keep,
            stride=2,
            pretrained=False,
            progress=True,
            device="cpu", 
            do_initial_max_pool=True,
        )


        
        self.output_projector = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), Squeeze(-1), Squeeze(-1), nn.LazyLinear(d_model), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model, out_dims))


    def forward(self, x):
        return self.output_projector((self.backbone(self.initial_rgb(x))))
