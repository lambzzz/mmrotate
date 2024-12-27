import torch.nn as nn
# from torchvision.models.swin_transformer import SwinTransformerBlock
from mmdet.models.backbones.swin import SwinBlock

class Swin(nn.Module):
    def __init__(self, window_size, shift_size = [0, 0]):
        super(Swin, self).__init__()
        # self.encoder = SwinTransformerBlock(dim=256, num_heads=8, window_size=window_size, shift_size=shift_size)
        self.encoder = SwinBlock(embed_dims=256, 
                                 num_heads=8, 
                                 feedforward_channels=1024, 
                                 window_size=window_size[0],)
        
    def forward(self, x):
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b, h*w, c)
        x = self.encoder(x, [h, w])
        x = x.reshape(b, h, w, c).permute(0, 3, 1, 2)
        return x