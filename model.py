import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F


def truncated_normal_(tensor, mean=0, std=1):
    with torch.no_grad():
        size = tensor.shape
        tmp = tensor.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std).add_(mean)
        return tensor

def init_weights(*modules):
    for module in modules:
        for m in module.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                truncated_normal_(m.weight, mean=0, std=1e-3)
                nn.init.constant_(m.bias, 0.0)

def kaiming_init_weights(*modules):
    for module in modules:
        for m in module.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0.0)

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, activate="lrelu"):
        super(Conv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.activate = activate

        self.conv = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding)

        # self.batch = nn.BatchNorm2d(self.out_channels)

        if activate == "lrelu":
            self.act = nn.LeakyReLU(0.2)
        elif activate == "tanh":
            self.act = nn.Tanh()
        elif activate == 'sigmoid':
            self.act = nn.Sigmoid()
        elif activate == 'relu':
            self.act = nn.ReLU()
        elif activate == 'prelu':
            self.act = nn.PReLU()
        else:
            self.act = None

        layers = filter(lambda x: x is not None, [self.conv, self.act])

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)

        return x

class Residual_Block(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=1, padding=1, activate='lrelu'):
        super(Residual_Block, self).__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.activate = activate

        self.conv1 = nn.Conv2d(self.in_channels, self.in_channels, self.kernel_size, self.stride, self.padding)
        self.conv2 = nn.Conv2d(self.in_channels, self.in_channels, self.kernel_size, self.stride, self.padding)

        if self.activate == 'relu':
            self.act1 = nn.ReLU(inplace=True)
        elif self.activate == 'lrelu':
            self.act1 = nn.LeakyReLU(negative_slope=0.2)
        elif self.activate == 'tanh':
            self.act1 = nn.Tanh()
        elif self.activate == 'sigmoid':
            self.act1 = nn.Sigmoid()
        elif self.activate == 'prelu':
            self.act1 = nn.PReLU()
            self.act2 = nn.PReLU()
        else:
            self.act1 = None

        layers = filter(lambda x: x is not None, [self.conv1, self.act1, self.conv2])

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        skip = x
        x = self.layers(x)

        if self.activate == 'prelu':
            return self.act2(x+skip)
        else:
            return self.act1(x+skip)

class Residual_Block_bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, activate='lrelu'):
        super(Residual_Block_bottleneck, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.activate = activate

        self.conv1 = nn.Conv2d(self.in_channels, self.in_channels * 4, self.kernel_size, self.stride, self.padding)
        self.conv2 = nn.Conv2d(self.in_channels * 4, self.in_channels * 4, self.kernel_size, self.stride, self.padding)
        self.conv3 = nn.Conv2d(self.in_channels * 4, self.out_channels, self.kernel_size, self.stride, self.padding)

        self.shortcut = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=self.stride, padding=0)

        if self.activate == 'relu':
            self.act1 = nn.ReLU(inplace=True)
        elif self.activate == 'lrelu':
            self.act1 = nn.LeakyReLU(negative_slope=0.2)
        elif self.activate == 'tanh':
            self.act1 = nn.Tanh()
        elif self.activate == 'sigmoid':
            self.act1 = nn.Sigmoid()
        elif self.activate == 'prelu':
            self.act1 = nn.PReLU()
            self.act2 = nn.PReLU()
        else:
            self.act1 = None

        layers = filter(lambda x: x is not None, [self.conv1, self.act1, self.conv2, self.act1, self.conv3])

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        skip = x
        x = self.layers(x)

        if self.activate == 'prelu':
            return self.act2(x + self.shortcut(skip))
        else:
            return self.act1(x + self.shortcut(skip))

class Upsampler(nn.Module):
    def __init__(self, in_channels):
        super(Upsampler, self).__init__()

        self.conv1 = Conv(in_channels=in_channels, out_channels=16)
        self.conv2 = Conv(in_channels=16, out_channels=in_channels)

        kaiming_init_weights(self.conv1, self.conv2)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bicubic')
        x = self.conv1(x)
        x = F.interpolate(x, scale_factor=2, mode='bicubic')
        x = self.conv2(x)

        return x

class SSConv(nn.Module):
    def __init__(self, in_channels, up):
        super(SSConv, self).__init__()
        self.up_size = up

        self.conv_up1 = nn.Conv2d(in_channels, out_channels=in_channels * up * up, kernel_size=3, stride=1, padding=1,
                                  bias=True)

    def mapping(self, x):
        B, C, H, W = x.shape
        C1, H1, W1 = C // (self.up_size * self.up_size), H * self.up_size, W * self.up_size
        x = x.reshape(B, C1, self.up_size, self.up_size, H, W)
        x = x.permute(0, 1, 4, 2, 5, 3)
        x = x.reshape(B, C1, H1, W1)
        return x

    def forward(self, x):
        x = self.conv_up1(x)
        return self.mapping(x)

class spatial_attention(nn.Module):
    def __init__(self, in_channels, out_channels=4):
        super(spatial_attention, self).__init__()

        self.conv1 = Conv(in_channels, out_channels)

        self.conv2 = Conv(in_channels=2, out_channels=1, kernel_size=5, padding=2, activate='sigmoid')

        kaiming_init_weights(self.conv1, self.conv2)

    def forward(self, x):
        out = self.conv1(x)
        max_pool = torch.max(out, 1, keepdim=True)[0]
        avg_pool = torch.mean(out, 1, keepdim=True)
        y = torch.cat([max_pool, avg_pool], 1)
        y1 = self.conv2(y)
        output = torch.mul(y1, out)
        return output

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(self, channels, kernel_size=3, reduction=16, bn=False, act=nn.LeakyReLU(), res_scale=1):
        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(Conv(channels, channels, kernel_size))
            if bn:
                modules_body.append(nn.BatchNorm2d(channels))
            if i == 0:
                modules_body.append(act)
        modules_body.append(CALayer(channels, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

class shading(nn.Module):
    def __init__(self, ms_channels, pan_channel=1):
        super(shading, self).__init__()

        self.ms_channels = ms_channels
        self.pan_channel = pan_channel

        if self.ms_channels == 4:
            self.channel1 = nn.Sequential(
                spatial_attention(in_channels=self.pan_channel, out_channels=16),
                Conv(in_channels=16, out_channels=64),
                Conv(in_channels=64, out_channels=16),
                Conv(in_channels=16, out_channels=self.ms_channels)
            )
            # self.channel1 = nn.Sequential(
            #     spatial_attention(in_channels=self.pan_channel, out_channels=64),
            #     Conv(in_channels=64, out_channels=self.ms_channels)
            # )

            self.channel1_conv2 = Conv(in_channels=self.ms_channels, out_channels=self.pan_channel, kernel_size=3)

            # self.channel2 = nn.Sequential(
            #     spatial_attention(in_channels=self.pan_channel, out_channels=16),
            #     Conv(in_channels=16, out_channels=32),
            #     Conv(in_channels=32, out_channels=16),
            #     Conv(in_channels=16, out_channels=self.ms_channels)
            # )

            self.channel2_conv2 = Conv(in_channels=self.ms_channels, out_channels=self.pan_channel, kernel_size=3)

            # self.channel3 = nn.Sequential(
            #     spatial_attention(in_channels=self.pan_channel, out_channels=16),
            #     Conv(in_channels=16, out_channels=32),
            #     Conv(in_channels=32, out_channels=16),
            #     Conv(in_channels=16, out_channels=self.ms_channels)
            # )

            self.channel3_conv2 = Conv(in_channels=self.ms_channels, out_channels=self.pan_channel, kernel_size=3)

            # self.channel4 = nn.Sequential(
            #     spatial_attention(in_channels=self.pan_channel, out_channels=16),
            #     Conv(in_channels=16, out_channels=32),
            #     Conv(in_channels=32, out_channels=16),
            #     Conv(in_channels=16, out_channels=self.ms_channels)
            # )

            self.channel4_conv2 = Conv(in_channels=self.ms_channels, out_channels=self.pan_channel, kernel_size=3)

            self.conv1 = Conv(in_channels=self.ms_channels, out_channels=self.ms_channels, kernel_size=1, padding=0)

            # kaiming_init_weights(self.channel1, self.channel2, self.channel3, self.channel4, self.conv1,
            #              self.channel1_conv2, self.channel2_conv2, self.channel3_conv2, self.channel4_conv2)

            kaiming_init_weights(self.channel1, self.channel1_conv2, self.channel2_conv2, self.channel3_conv2,
                                 self.channel4_conv2, self.conv1)

        elif self.ms_channels == 8:
            self.channel1 = nn.Sequential(
                spatial_attention(in_channels=self.pan_channel, out_channels=16),
                Conv(in_channels=16, out_channels=64),
                Conv(in_channels=64, out_channels=16),
                Conv(in_channels=16, out_channels=self.ms_channels)
            )
            # self.channel1 = Conv(in_channels=self.pan_channel, out_channels=self.ms_channels, kernel_size=1,padding=0)

            # self.channel2 = nn.Sequential(
            #     # Conv(in_channels=self.pan_channel, out_channels=self.ms_channels)
            #     spatial_attention(in_channels=self.pan_channel, out_channels=16),
            #     Conv(in_channels=16, out_channels=32),
            #     Conv(in_channels=32, out_channels=16),
            #     Conv(in_channels=16, out_channels=self.ms_channels)
            # )
            #
            # self.channel3 = nn.Sequential(
            #     # Conv(in_channels=self.pan_channel, out_channels=self.ms_channels)
            #     spatial_attention(in_channels=self.pan_channel, out_channels=16),
            #     Conv(in_channels=16, out_channels=32),
            #     Conv(in_channels=32, out_channels=16),
            #     Conv(in_channels=16, out_channels=self.ms_channels)
            # )
            #
            # self.channel4 = nn.Sequential(
            #     # Conv(in_channels=self.pan_channel, out_channels=self.ms_channels)
            #     spatial_attention(in_channels=self.pan_channel, out_channels=16),
            #     Conv(in_channels=16, out_channels=32),
            #     Conv(in_channels=32, out_channels=16),
            #     Conv(in_channels=16, out_channels=self.ms_channels)
            # )
            #
            # self.channel5 = nn.Sequential(
            #     # Conv(in_channels=self.pan_channel, out_channels=self.ms_channels)
            #     spatial_attention(in_channels=self.pan_channel, out_channels=16),
            #     Conv(in_channels=16, out_channels=32),
            #     Conv(in_channels=32, out_channels=16),
            #     Conv(in_channels=16, out_channels=self.ms_channels)
            # )
            #
            # self.channel6 = nn.Sequential(
            #     # Conv(in_channels=self.pan_channel, out_channels=self.ms_channels)
            #     spatial_attention(in_channels=self.pan_channel, out_channels=16),
            #     Conv(in_channels=16, out_channels=32),
            #     Conv(in_channels=32, out_channels=16),
            #     Conv(in_channels=16, out_channels=self.ms_channels)
            # )
            #
            # self.channel7 = nn.Sequential(
            #     # Conv(in_channels=self.pan_channel, out_channels=self.ms_channels)
            #     spatial_attention(in_channels=self.pan_channel, out_channels=16),
            #     Conv(in_channels=16, out_channels=32),
            #     Conv(in_channels=32, out_channels=16),
            #     Conv(in_channels=16, out_channels=self.ms_channels)
            # )
            #
            # self.channel8 = nn.Sequential(
            #     # Conv(in_channels=self.pan_channel, out_channels=self.ms_channels)
            #     spatial_attention(in_channels=self.pan_channel, out_channels=16),
            #     Conv(in_channels=16, out_channels=32),
            #     Conv(in_channels=32, out_channels=16),
            #     Conv(in_channels=16, out_channels=self.ms_channels)
            # )

            self.channel1_conv2 = Conv(in_channels=self.ms_channels, out_channels=self.pan_channel, kernel_size=3)

            self.channel2_conv2 = Conv(in_channels=self.ms_channels, out_channels=self.pan_channel, kernel_size=3)

            self.channel3_conv2 = Conv(in_channels=self.ms_channels, out_channels=self.pan_channel, kernel_size=3)

            self.channel4_conv2 = Conv(in_channels=self.ms_channels, out_channels=self.pan_channel, kernel_size=3)

            self.channel5_conv2 = Conv(in_channels=self.ms_channels, out_channels=self.pan_channel, kernel_size=3)

            self.channel6_conv2 = Conv(in_channels=self.ms_channels, out_channels=self.pan_channel, kernel_size=3)

            self.channel7_conv2 = Conv(in_channels=self.ms_channels, out_channels=self.pan_channel, kernel_size=3)

            self.channel8_conv2 = Conv(in_channels=self.ms_channels, out_channels=self.pan_channel, kernel_size=3)

            self.conv1 = Conv(in_channels=self.ms_channels, out_channels=self.ms_channels, kernel_size=1, padding=0)

            # kaiming_init_weights(self.channel1, self.channel2, self.channel3, self.channel4, self.conv1,
            #              self.channel5, self.channel6, self.channel7, self.channel8,
            #              self.channel1_conv2, self.channel2_conv2, self.channel3_conv2, self.channel4_conv2,
            #              self.channel5_conv2, self.channel6_conv2, self.channel7_conv2, self.channel8_conv2)

            kaiming_init_weights(self.channel1, self.channel1_conv2, self.channel2_conv2, self.channel3_conv2,
                                 self.channel4_conv2, self.channel5_conv2, self.channel6_conv2, self.channel7_conv2,
                                 self.channel8_conv2, self.conv1)

            # kaiming_init_weights(self.conv1)
            
        else:
            self.channel1 = nn.Sequential(
                spatial_attention(in_channels=self.pan_channel, out_channels=16),
                Conv(in_channels=16, out_channels=64),
                Conv(in_channels=64, out_channels=16),
                Conv(in_channels=16, out_channels=self.ms_channels)
            )

            self.conv1 = Conv(in_channels=self.ms_channels, out_channels=self.ms_channels, kernel_size=1, padding=0)
            
            kaiming_init_weights(self.channel1, self.conv1)

    def forward(self, pan):
        if self.ms_channels == 4:
            # channel1 = self.channel1_conv2(self.channel1(pan[:, 0, :, :].unsqueeze(1)) + pan[:, 0, :, :].unsqueeze(1))
            # channel2 = self.channel2_conv2(self.channel1(pan[:, 1, :, :].unsqueeze(1)) + pan[:, 1, :, :].unsqueeze(1))
            # channel3 = self.channel3_conv2(self.channel1(pan[:, 2, :, :].unsqueeze(1)) + pan[:, 2, :, :].unsqueeze(1))
            # channel4 = self.channel4_conv2(self.channel1(pan[:, 3, :, :].unsqueeze(1)) + pan[:, 3, :, :].unsqueeze(1))

            channel1 = self.channel1_conv2(self.channel1(pan[:, 0, :, :].unsqueeze(1)))
            channel2 = self.channel2_conv2(self.channel1(pan[:, 1, :, :].unsqueeze(1)))
            channel3 = self.channel3_conv2(self.channel1(pan[:, 2, :, :].unsqueeze(1)))
            channel4 = self.channel4_conv2(self.channel1(pan[:, 3, :, :].unsqueeze(1)))

            # channel1 = self.channel1_conv2(self.channel1(pan[:, 0, :, :].unsqueeze(1)))
            # channel2 = self.channel2_conv2(self.channel2(pan[:, 1, :, :].unsqueeze(1)))
            # channel3 = self.channel3_conv2(self.channel3(pan[:, 2, :, :].unsqueeze(1)))
            # channel4 = self.channel4_conv2(self.channel4(pan[:, 3, :, :].unsqueeze(1)))

            return self.conv1(torch.cat([channel1, channel2, channel3, channel4], 1))

        elif self.ms_channels == 8:
            channel1 = self.channel1_conv2(self.channel1(pan[:, 0, :, :].unsqueeze(1)))
            channel2 = self.channel2_conv2(self.channel1(pan[:, 1, :, :].unsqueeze(1)))
            channel3 = self.channel3_conv2(self.channel1(pan[:, 2, :, :].unsqueeze(1)))
            channel4 = self.channel4_conv2(self.channel1(pan[:, 3, :, :].unsqueeze(1)))
            channel5 = self.channel5_conv2(self.channel1(pan[:, 4, :, :].unsqueeze(1)))
            channel6 = self.channel6_conv2(self.channel1(pan[:, 5, :, :].unsqueeze(1)))
            channel7 = self.channel7_conv2(self.channel1(pan[:, 6, :, :].unsqueeze(1)))
            channel8 = self.channel8_conv2(self.channel1(pan[:, 7, :, :].unsqueeze(1)))

            # channel1 = self.channel1_conv2(self.channel1(pan[:, 0, :, :].unsqueeze(1)))
            # channel2 = self.channel2_conv2(self.channel2(pan[:, 1, :, :].unsqueeze(1)))
            # channel3 = self.channel3_conv2(self.channel3(pan[:, 2, :, :].unsqueeze(1)))
            # channel4 = self.channel4_conv2(self.channel4(pan[:, 3, :, :].unsqueeze(1)))
            # channel5 = self.channel5_conv2(self.channel5(pan[:, 4, :, :].unsqueeze(1)))
            # channel6 = self.channel6_conv2(self.channel6(pan[:, 5, :, :].unsqueeze(1)))
            # channel7 = self.channel7_conv2(self.channel7(pan[:, 6, :, :].unsqueeze(1)))
            # channel8 = self.channel8_conv2(self.channel8(pan[:, 7, :, :].unsqueeze(1)))

            return self.conv1(torch.cat([channel1, channel2, channel3, channel4, channel5, channel6, channel7, channel8], 1))

            # return self.conv1(pan)

        else:
            channel = []
            for i in range(self.ms_channels):
                channel.append(self.channel1(pan[:, i, :, :].unsqueeze(1)))

            return self.conv1(torch.cat(channel, 1))

class reflectance(nn.Module):
    def __init__(self, ms_channels, pan_channel=1):
        super(reflectance, self).__init__()

        self.ms_channels = ms_channels
        self.pan_channel = pan_channel

        # self.ps = spatial_attention(in_channels=self.ms_channels, out_channels=self.ms_channels * 16)
        self.ps = Conv(in_channels=self.ms_channels, out_channels=self.ms_channels * 16)

        self.rcab = nn.Sequential(
            spatial_attention(in_channels=self.ms_channels * 16, out_channels=self.ms_channels * 16),
            RCAB(channels=self.ms_channels * 16, bn=True),
            RCAB(channels=self.ms_channels * 16, bn=True),
            Conv(in_channels=self.ms_channels * 16, out_channels=self.ms_channels * 16)
        )

        # self.rcab = nn.Sequential(
        #     Conv(in_channels=64, out_channels=64),
        #     RCAB(channels=64, bn=True),
        #     spatial_attention(in_channels=64, out_channels=64),
        # )

        self.conv1 = Conv(in_channels=self.ms_channels * 16, out_channels=self.ms_channels, kernel_size=1, padding=0)

        kaiming_init_weights(self.ps, self.rcab, self.conv1)
        # kaiming_init_weights(self.ps, self.conv1)

    def forward(self, ms):
        ps = self.ps(ms)

        rcab = self.rcab(ps)

        out = self.conv1(ps + rcab)

        return out

class MyNet(nn.Module):
    def __init__(self, ms_channels, pan_channel=1):
        super(MyNet, self).__init__()

        self.ms_channels = ms_channels
        self.pan_channel = pan_channel

        # self.pre_fusion = nn.Sequential(
        #     Residual_Block_bottleneck(in_channels=self.ms_channels + self.pan_channel, out_channels=64),
        #     Residual_Block(in_channels=64),
        #     Residual_Block(in_channels=64),
        #     Residual_Block_bottleneck(in_channels=64, out_channels=self.ms_channels)
        # )

        self.pan_conv1 = Conv(in_channels=self.pan_channel, out_channels=self.ms_channels, kernel_size=3, padding=1)

        # shade = []
        # for i in range(4):
        #     shade.append(shading(ms_channels=self.ms_channels))
        # self.shade = nn.Sequential(*shade)

        self.shade1 = shading(ms_channels=self.ms_channels)
        self.shade2 = shading(ms_channels=self.ms_channels)
        self.shade3 = shading(ms_channels=self.ms_channels)

        self.pan_conv2 = nn.Sequential(
            Conv(in_channels=self.ms_channels * 3, out_channels=self.ms_channels),
            Conv(in_channels=self.ms_channels, out_channels=self.ms_channels, kernel_size=1, padding=0, activate='tanh')
        )
            
        self.ms_conv1 = Conv(in_channels=self.ms_channels, out_channels=self.ms_channels, kernel_size=3, padding=1)

        self.ref1 = reflectance(ms_channels=self.ms_channels)
        self.ref2 = reflectance(ms_channels=self.ms_channels)
        self.ref3 = reflectance(ms_channels=self.ms_channels)

        self.ms_conv2 = nn.Sequential(
            Conv(in_channels=self.ms_channels * 3, out_channels=self.ms_channels),
            Conv(in_channels=self.ms_channels, out_channels=self.ms_channels, kernel_size=1, padding=0, activate='tanh')
        )

        self.out_conv = Conv(in_channels=self.ms_channels, out_channels=self.ms_channels, kernel_size=1, padding=0)

        kaiming_init_weights(self.pan_conv1, self.pan_conv2, self.ms_conv2, self.ms_conv1, self.out_conv)

    def forward(self, ms, pan):
        ms_4 = F.interpolate(ms, scale_factor=4, mode='bicubic')

        fusion = self.pan_conv1(pan)

        shade1 = self.shade1(fusion)
        shade2 = self.shade2(shade1 + fusion)
        # shade3 = self.shade3(shade2 + shade1 + fusion)
        # shade = self.pan_conv2(torch.cat([shade1, shade2, shade3, fusion], 1))

        # shade = self.pan_conv2(torch.cat([shade1, fusion], 1))

        shade = self.pan_conv2(torch.cat([shade1, shade2, fusion], 1))

        # shade = self.pan_conv2(fusion)

        fusion_ref = self.ms_conv1(ms_4)

        ms_ref1 = self.ref1(fusion_ref)
        ms_ref2 = self.ref2(ms_ref1 + fusion_ref)
        # ms_ref3 = self.ref3(ms_ref2 + ms_ref1 + fusion_ref)
        # ref = self.ms_conv2(torch.cat([ms_ref1, ms_ref2, ms_ref3, fusion_ref], 1))

        # ref = self.ms_conv2(torch.cat([ms_ref1, fusion_ref], 1))

        ref = self.ms_conv2(torch.cat([ms_ref1, ms_ref2, fusion_ref], 1))

        # ref = self.ms_conv2(fusion_ref)

        out = self.out_conv(shade * ref)

        # out = shade * ref

        return out, shade, ref

if __name__ == "__main__":

    net=MyNet(ms_channels=8).cuda()
    total = sum(p.numel() for p in net.parameters())
    print(net)
    print(f'{total / 1e6:.4f}Mb')