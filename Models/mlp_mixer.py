import torch
import numpy as np
from torch import nn
from einops.layers.torch import Rearrange


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class MixerBlock(nn.Module):

    def __init__(self, dim, num_patch, token_dim, channel_dim, dropout = 0.):
        super().__init__()
        
        self.token_mixing = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),
            FeedForward(num_patch, token_dim, dropout),
            Rearrange('b d n -> b n d')
        )

        self.channel_mixing = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, channel_dim, dropout),
        )

    def forward(self, x):

        x = x + self.token_mixing(x)

        x = x + self.channel_mixing(x)

        return x


class MLPMixer(nn.Module):

    def __init__(self, in_channels, dim, num_classes, patch_size, image_size, depth, token_dim, channel_dim):
        super().__init__()
        # dim is number of channels - C
        # channel_dim is the hidden dimension of the channel MLP
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        self.num_patch =  (image_size// patch_size) ** 2  #num_patch is number of tokens 
        #token_dim is the hidden dimension of the token MLP
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dim, patch_size, patch_size),
            Rearrange('b c h w -> b (h w) c'),
        )

        self.MixerBlock = nn.ModuleList([])

        for _ in range(depth):
            self.MixerBlock.append(MixerBlock(dim, self.num_patch, token_dim, channel_dim))

        self.pre_head_layer_norm = nn.LayerNorm(dim)

        self.head = nn.Sequential(
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):


        x = self.stem(x)

        for mixer_block in self.MixerBlock:
            x = mixer_block(x)

        x = self.pre_head_layer_norm(x)

        x = x.mean(dim=1)

        return self.head(x)




if __name__ == "__main__":
    img = torch.ones([1, 3, 224, 224])

    model = MLPMixer(in_channels=3, image_size=224, patch_size=16, num_classes=1000,
                     dim=512, depth=8, token_dim=256, channel_dim=2048)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)

    out_img = model(img)

    print("Shape of out :", out_img.shape)  # [B, in_channels, image_size, image_size]