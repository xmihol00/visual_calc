from torch import nn

def CNN_downsampling(in_channels, out_channels):
    intermidiate_channels = out_channels * 2
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, (3, 3), stride=1, padding=0),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.1),
        nn.Conv2d(out_channels, intermidiate_channels, (3, 3), stride=2, padding=1),
        nn.BatchNorm2d(intermidiate_channels),
        nn.LeakyReLU(0.1),
        nn.Conv2d(intermidiate_channels, out_channels, (1, 1), stride=1, padding=0),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.1)
    )

def CNN_residual(channels):
    intermidiate_channels = channels * 2
    return nn.Sequential(
        nn.Conv2d(channels, intermidiate_channels, (3, 3), stride=1, padding=1),
        nn.BatchNorm2d(intermidiate_channels),
        nn.LeakyReLU(0.1),
        nn.Conv2d(intermidiate_channels, channels, (1, 1), stride=1, padding=0),
        nn.BatchNorm2d(channels),
        nn.LeakyReLU(0.1),
    )

def YOLO(in_channels, out_channels):
    intermidiate_channels = in_channels * 2
    return nn.Sequential(
        nn.Conv2d(in_channels, intermidiate_channels, (3, 3), stride=1, padding=0),
        nn.BatchNorm2d(intermidiate_channels),
        nn.LeakyReLU(0.1),
        nn.Conv2d(intermidiate_channels, out_channels, (1, 1), stride=1, padding=0),
    )
