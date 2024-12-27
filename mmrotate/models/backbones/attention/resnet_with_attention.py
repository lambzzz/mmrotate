import torch
from mmdet.models.backbones import ResNet
from .coordattention import CoordAtt
from .seattention import SEAttention
from ...builder import ROTATED_BACKBONES
 
 
# 定义带attention的resnet18基类
class ResNetWithAttention(ResNet):
    def __init__(self , **kwargs):
        super(ResNetWithAttention, self).__init__(**kwargs)
        # 目前将注意力模块加在最后的三个输出特征层
        # resnet输出四个特征层
        if self.depth in (18, 34):
            self.dims = (64, 128, 256, 512)
        elif self.depth in (50, 101, 152):
            self.dims = (256, 512, 1024, 2048)
        else:
            raise Exception()
        self.attention1 = self.get_attention_module(self.dims[1])     
        self.attention2 = self.get_attention_module(self.dims[2])     
        self.attention3 = self.get_attention_module(self.dims[3])     
    
    # 子类只需要实现该attention即可
    def get_attention_module(self, dim):
        raise NotImplementedError()
    
    def forward(self, x):
        outs = super().forward(x)
        outs = list(outs)
        outs[1] = self.attention1(outs[1])
        outs[2] = self.attention2(outs[2])
        outs[3] = self.attention3(outs[3])    
        outs = tuple(outs)
        return outs
    
@ROTATED_BACKBONES.register_module()
class ResNetWithCA(ResNetWithAttention):
    def __init__(self , **kwargs):
        super(ResNetWithCA, self).__init__(**kwargs)
 
    # 子类只需要实现该attention即可
    def get_attention_module(self, dim):
        return CoordAtt(inp=dim, oup=dim, reduction=32)
    
@ROTATED_BACKBONES.register_module()
class ResNetWithSE(ResNetWithAttention):
    def __init__(self , **kwargs):
        super(ResNetWithSE, self).__init__(**kwargs)
 
    # 子类只需要实现该attention即可
    def get_attention_module(self, dim):
        return SEAttention(channel=dim, reduction=16)
 
 
if __name__ == "__main__":
    # model = ResNet(depth=18)
    # model = ResNet(depth=34)
    # model = ResNet(depth=50)
    # model = ResNet(depth=101)    
    # model = ResNet(depth=152)
    # model = ResNetWithCoordAttention(depth=18)
    model = ResNetWithSE(depth=18)
    x = torch.rand(1, 3, 224, 224)
    outs = model(x)
    # print(outs.shape)
    for i, out in enumerate(outs):
        print(i, out.shape)