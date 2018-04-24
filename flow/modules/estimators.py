
import torch
import torch.nn as nn


def conv(batch_norm, in_planes, out_planes, kernel_size=3, stride=1):
    if batch_norm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                      stride=stride, padding=(kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                      stride=stride, padding=(kernel_size - 1) // 2, bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )


def soft_deconv(in_planes, out_planes, upsample_mode='bilinear'):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode=upsample_mode),
        nn.Conv2d(in_planes, out_planes, kernel_size=3,
                  stride=1, padding=1, bias=True), # TURN OFF BIAS??
        nn.LeakyReLU(0.1, inplace=True)
    )


def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4,
                           stride=2, padding=1, bias=True),
        nn.LeakyReLU(0.1, inplace=True)
    )


def soft_conv_transpose(in_planes, out_planes, kernel_size=3, stride=2, padding=1, bias=False):
    return nn.Sequential(
        nn.Upsample(scale_factor=stride, mode='bilinear'),
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                  stride=1, padding=1, bias=False)
    )


def predict_flow(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)


class ConvDeconvEstimator(nn.Module):
    """
        The output channel size is the same as the input.
        This is done by removing the last two convolutional
        layers of FlowNetS, and adding two deconvolutional 
        layers at the end.
    """

    def __init__(self, input_channels=4, output_channels=2, batch_norm=True, upsample_mode='bilinear'):
        super(ConvDeconvEstimator, self).__init__()

        self.batch_norm = batch_norm
        self.upsample_mode = upsample_mode
        self.input_channels = input_channels
        self.conv1 = conv(self.batch_norm, input_channels, 64, kernel_size=3, stride=2)
        self.conv2 = conv(self.batch_norm, 64, 128, kernel_size=3, stride=2)
        self.conv3 = conv(self.batch_norm, 128, 256, kernel_size=3, stride=2)
        self.conv3_1 = conv(self.batch_norm, 256, 256, kernel_size=3)
        self.conv4 = conv(self.batch_norm, 256, 512, kernel_size=3, stride=2)
        self.conv4_1 = conv(self.batch_norm, 512, 512, kernel_size=3)
        self.conv5 = conv(self.batch_norm, 512, 1024, stride=2)
        self.conv5_1 = conv(self.batch_norm, 1024, 1024)

        if upsample_mode == 'deconv':
            self.deconv4 = deconv(1024, 256)
            self.deconv3 = deconv(768, 128)
            self.deconv2 = deconv(384, 64)
            self.deconv1 = deconv(192, 32)
            self.deconv0 = deconv(96, 16)
        else:
            self.deconv4 = soft_deconv(1024, 256, upsample_mode=upsample_mode)
            self.deconv3 = soft_deconv(768, 128, upsample_mode=upsample_mode)
            self.deconv2 = soft_deconv(384, 64, upsample_mode=upsample_mode)
            self.deconv1 = soft_deconv(192, 32, upsample_mode=upsample_mode)
            self.deconv0 = soft_deconv(96, 16, upsample_mode=upsample_mode)

        self.predict_flow0 = predict_flow(16 + input_channels, output_channels)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.02 / n)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):

        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))

        out_deconv4 = self.deconv4(out_conv5)
        concat4 = torch.cat((out_conv4, out_deconv4), 1)
        out_deconv3 = self.deconv3(concat4)
        concat3 = torch.cat((out_conv3, out_deconv3), 1)
        out_deconv2 = self.deconv2(concat3)
        concat2 = torch.cat((out_conv2, out_deconv2), 1)
        out_deconv1 = self.deconv1(concat2)
        concat1 = torch.cat((out_conv1, out_deconv1), 1)
        out_deconv0 = self.deconv0(concat1)
        concat0 = torch.cat((x, out_deconv0), 1)
        flow0 = self.predict_flow0(concat0)

        return flow0
