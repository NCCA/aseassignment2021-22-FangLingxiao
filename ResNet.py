import torch.nn as nn
import torch

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()
        self.downsample = downsample

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=(2 if downsample else 1), padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if downsample:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()  # use identity mapping

    def forward(self, inp):
        shortcut = self.shortcut(inp)

        # A new variable out is used to avoid in-place modification of inp. 
        # Doing this ensures that no errors will occur due to in-place modifications when performing gradient calculations
        out = self.conv1(inp)
        out = self.bn1(out)
        out = nn.ReLU()(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = nn.ReLU()(out)

        out = out + shortcut # dont use += to avoid in-place!!!!!(It took me forever to find out the reason for the error)
        out = nn.ReLU()(out)
        return out
        

class ResNet(nn.Module):
    def __init__(self, in_channels, resblock, outputs):
        super().__init__()
        self.layer0_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.layer0_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer0_bn   = nn.BatchNorm2d(64)
        self.layer0_relu = nn.ReLU()

        self.layer1_res1 = resblock(64, 64, downsample=False)
        self.layer1_res2 = resblock(64, 64, downsample=False)

        self.layer2_res1 = resblock(64, 128, downsample=True)
        self.layer2_res2 = resblock(128, 128, downsample=False)

        self.layer3_res1 = resblock(128, 256, downsample=True)
        self.layer3_res2 = resblock(256, 256, downsample=False)

        self.layer4_res1 = resblock(256, 512, downsample=True)
        self.layer4_res2 = resblock(512, 512, downsample=False)

        self.gap         = nn.AdaptiveAvgPool2d(1)
        self.flat        = nn.Flatten() 
        self.fc          = nn.Linear(512, outputs)

    def forward(self, inp):
        inp = self.layer0_conv(inp)
        inp = self.layer0_pool(inp)
        inp = self.layer0_bn(inp)
        inp = self.layer0_relu(inp)
        
        inp = self.layer1_res1(inp)
        inp = self.layer1_res2(inp)
        
        inp = self.layer2_res1(inp)
        inp = self.layer2_res2(inp)
        
        inp = self.layer3_res1(inp)
        inp = self.layer3_res2(inp)
        
        inp = self.layer4_res1(inp)
        inp = self.layer4_res2(inp)
            
        inp = self.gap(inp)
        inp = self.flat(inp)
        inp = self.fc(inp)

        return inp
            
# convenience function
def get_resnet():
    return ResNet(1, ResBlock, outputs=10)
    

if __name__ == '__main__':
    tensor = torch.rand([1, 1, 128, 1000])
    model = get_resnet()                                          # Instantiate the ResNet model

    total_params = sum(p.numel() for p in model.parameters())     # Calculate the total number of parameters
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(                                 # Only the number of trainable parameters is counted.
        p.numel() for p in model.parameters() if p.requires_grad) # p.requires_grad indicates whether the parameter participates in gradient update
    print(f"{total_trainable_params:,} training parameters.")

    output = model(tensor)
    print(f"{output.shape:}Output feature size.")

