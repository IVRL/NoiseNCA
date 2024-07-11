import torch


def depthwise_conv(x, filters, padding='circular'):
    """filters: [filter_n, h, w]"""
    b, ch, h, w = x.shape
    y = x.reshape(b * ch, 1, h, w)
    y = torch.nn.functional.pad(y, [1, 1, 1, 1], padding)
    y = torch.nn.functional.conv2d(y, filters[:, None])
    return y.reshape(b, -1, h, w)


class NCA(torch.nn.Module):
    def __init__(self, chn, fc_dim, padding='circular', noise_level=0.0, cond_chn=0, device=None):
        super(NCA, self).__init__()
        self.device = device
        self.chn = chn
        self.fc_dim = fc_dim
        self.padding = padding
        self.cond_chn = cond_chn

        self.w1 = torch.nn.Conv2d(chn * 4 + cond_chn, fc_dim, 1, bias=True, device=device)
        self.w2 = torch.nn.Conv2d(fc_dim, chn, 1, bias=False, device=device)

        torch.nn.init.xavier_normal_(self.w1.weight, gain=0.2)
        torch.nn.init.zeros_(self.w2.weight)

        ident = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]], device=device)
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=device)
        lap_x = torch.tensor([[0.5, 0.0, 0.5], [2.0, -6.0, 2.0], [0.5, 0.0, 0.5]], device=device)

        self.register_buffer("noise_level", torch.tensor([noise_level], device=device))
        self.register_buffer("filters", torch.stack([ident, sobel_x, sobel_x.T, lap_x, lap_x.T]))

    def perception(self, s, dx=1.0, dy=1.0):
        """

        :param s: Cell state tensor of shape [b, chn, h, w]
        :param dx: Either a float or a tensor of shape [b, h, w]
        :param dy: Either a float or a tensor of shape [b, h, w]

        dx, dy are used to scale the sobel and laplacian filters.
        dx < 1.0 means that the patterns are gonna get stretched horizontally.
        dx > 1.0 means that the patterns are gonna get squeezed horizontally.
        """


device = torch.device("cpu")
nca = NCA(12, 96, device)
nca.noise_level
