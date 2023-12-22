import torch


class SEFNet(torch.nn.Module):

    def __init__(self, in_channels=3):
        super(SEFNet, self).__init__()
        self.c0 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 32, 3, 1),
            torch.nn.Conv2d(32, 32, 3, 1),
        )
        self.c1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 32, 3, 2),
            torch.nn.Conv2d(32, 32, 3, 2),
        )
        self.c2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 32, 3, 2),
            torch.nn.Conv2d(32, 32, 3, 3),
        )
        self.c3 = torch.nn.Conv2d(in_channels, 32, 1, 1)

    def forward(self, x):
        a = self.c0(x)
        b = self.c1(x)
        c = self.c2(x)
        d = self.c3(x)

        print(a.shape, b.shape, c.shape, d.shape)
        return


if __name__ == '__main__':
    model = SEFNet()
    x = torch.rand([1,3,224,224], dtype=torch.float32)
    p = model(x)

    print(p.shape)