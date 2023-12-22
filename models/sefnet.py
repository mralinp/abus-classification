import torch


class SEFNet(torch.nn.Module):

    def __init__(self, in_channels=3):
        super(SEFNet, self).__init__()
        self.c0 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 32, 3, dilation=1, padding=(1,1)),
            torch.nn.Conv2d(32, 32, 3, dilation=1, padding=(1,1)),
        )
        self.c1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 32, 3, dilation=2, padding=(2,2)),
            torch.nn.Conv2d(32, 32, 3, dilation=2, padding=(2,2)),
        )
        self.c2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 32, 3, dilation=2, padding=(2,2)),
            torch.nn.Conv2d(32, 32, 3, dilation=3, padding=(3,3)),
        )
        self.c3 = torch.nn.Conv2d(in_channels, 32, 1, dilation=1)

        self.c4 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 128, 3),
            torch.nn.Conv2d(128, 128, 3),
        )

        self.c5 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 128, 3),
            torch.nn.Conv2d(128, 256, 3),
        )

    def forward(self, x):
        a = self.c0(x)
        b = self.c1(x)
        c = self.c2(x)
        d = self.c3(x)
        x = torch.cat((a,b,c,d), dim=1)
        x = torch.nn.functional.avg_pool2d(x, 2)
        x = self.c4(x)
        x = torch.nn.functional.avg_pool2d(x, 2)
        x = self.c5(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        x = x.squeeze(2)
        return x


if __name__ == '__main__':
    model = SEFNet()
    x = torch.rand([1, 3, 224, 224], dtype=torch.float32)
    p = model(x)
    print(p.shape)