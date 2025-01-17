
import torch
import torch.nn as nn

class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, x):
        s = self.conv(x)
        p = self.pool(s)
        return s, p

class attention_gate(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.Wg = nn.Conv2d(in_c[0], out_c // 2, kernel_size=1, padding=0)
        self.Wx = nn.Conv2d(in_c[1], out_c // 2, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.psi = nn.Sequential(
            nn.Conv2d(out_c // 2, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, g, x):
        g1 = self.Wg(g)
        x1 = self.Wx(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c[0], in_c[0], kernel_size=2, stride=2)
        self.ag = attention_gate(in_c, out_c)
        self.conv = conv_block(in_c[0] + out_c, out_c)

    def forward(self, g, x):
        g = self.up(g)
        x = self.ag(g, x)
        g = torch.cat([g, x], dim=1)
        return self.conv(g)

class AttentionUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder1 = encoder_block(3, 8)
        self.encoder2 = encoder_block(8, 16)
        self.encoder3 = encoder_block(16, 32)

        self.bottleneck = conv_block(32, 64)

        self.decoder1 = decoder_block([64, 32], 32)
        self.decoder2 = decoder_block([32, 16], 16)
        self.decoder3 = decoder_block([16, 8], 8)

        self.final_conv = nn.Conv2d(8, 1, kernel_size=1, padding=0)

    def forward(self, x):
        enc1, pool1 = self.encoder1(x)
        enc2, pool2 = self.encoder2(pool1)
        enc3, pool3 = self.encoder3(pool2)

        bottleneck = self.bottleneck(pool3)

        dec1 = self.decoder1(bottleneck, enc3)
        dec2 = self.decoder2(dec1, enc2)
        dec3 = self.decoder3(dec2, enc1)

        return self.final_conv(dec3)
