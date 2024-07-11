import torch


def depthwise_conv(x, filters, padding='circular'):
    """filters: [filter_n, h, w]"""
    b, ch, h, w = x.shape
    y = x.reshape(b * ch, 1, h, w)
    y = torch.nn.functional.pad(y, [1, 1, 1, 1], padding)
    y = torch.nn.functional.conv2d(y, filters[:, None])
    return y.reshape(b, -1, h, w)


class NCA(torch.nn.Module):
    """
    Base class for Neural Cellular Automata

    chn: Number of channels in the cell state
    fc_dim: Number of channels in the update MLP hidden layer
    padding: Padding mode for the perception (Convolution kernels)
    cond_chn: Number of conditional channels. For example a 2D positional encoding will add 2 extra condition channels.
    noise_level: Noise level for the seed initialization. 0.0 means that the seed is initialized with zeros.
                 noise_level = 1.0 means that the seed is initialized with uniform noise in [-0.5, 0.5].
    update_prob: Probability of updating a cell state in each iteration.
                 If update_prob = 1.0, all the cells are updated in each iteration.
    device: PyTorch device

    """

    def __init__(self, chn, fc_dim,
                 padding='circular', cond_chn=0,
                 noise_level=0.0, update_prob=0.5,
                 device=None):
        super(NCA, self).__init__()
        self.chn = chn
        self.fc_dim = fc_dim
        self.padding = padding
        self.cond_chn = cond_chn
        self.update_prob = update_prob

        self.device = device

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
        :param s: Cell states tensor of shape [b, chn, h, w]
        :param dx: Either a float or a tensor of shape [b, h, w]
        :param dy: Either a float or a tensor of shape [b, h, w]

        dx, dy are used to scale the sobel and laplacian filters.
        dx, dy < 1.0 means that the patterns are gonna get stretched horizontally, vertically.
        dx, dy > 1.0 means that the patterns are gonna get squeezed horizontally, vertically.
        """

        def transform(x):
            # This function merges the lap_x and lap_y into a single laplacian filter
            b, c, h, w = x.shape
            x = torch.stack([
                x[:, ::5],
                x[:, 1::5],
                x[:, 2::5],
                x[:, 3::5] + x[:, 4::5]
            ],
                dim=2)  # [b, chn, 4, h, w]
            return x.reshape(b, -1, h, w)  # [b, 4 * chn, h, w]

        z = depthwise_conv(s, self.filters, self.padding)  # [b, 5 * chn, h, w]
        if dx == 1.0 and dy == 1.0:
            return transform(z)

        if not isinstance(dx, torch.Tensor) or dx.ndim != 3:
            dx = torch.tensor([dx], device=s.device)[:, None, None]  # [1, 1, 1]
        if not isinstance(dy, torch.Tensor) or dy.ndim != 3:
            dy = torch.tensor([dy], device=s.device)[:, None, None]  # [1, 1, 1]

        scale = 1.0 / torch.stack([torch.ones_like(dx), dx, dy, dx ** 2, dy ** 2], dim=1)
        scale = torch.tile(scale, (1, self.chn, 1, 1))
        z = z * scale
        # z[:, ::5] the identity filter is not scaled
        # z[:, 1::5] = z[:, 1::5] * dx (resembling 1st order derivative in x direction)
        # z[:, 2::5] = z[:, 2::5] * dy (resembling 1st order derivative in y direction)
        # z[:, 3::5] = z[:, 3::5] * dx ** 2 (resembling 2nd order derivative in x direction)
        # z[:, 4::5] = z[:, 4::5] * dy ** 2 (resembling 2nd order derivative in y direction)
        return transform(z)

    def adaptation(self, s, dx=1.0, dy=1.0):
        z = self.perception(s, dx, dy)
        print(z.shape)
        delta_s = self.w2(torch.relu(self.w1(z)))
        return delta_s

    def step_euler(self, s, dx=1.0, dy=1.0, dt=1.0):
        delta_s = self.adaptation(s, dx, dy)
        M = 1.0
        if self.update_prob < 1.0:
            b, _, h, w = s.shape
            M = (torch.rand(b, 1, h, w, device=s.device) + self.update_prob).floor()

        return s + delta_s * M * dt

    def step_rk4(self, s, dx=1.0, dy=1.0, dt=1.0):
        M = 1.0
        if self.update_prob < 1.0:
            b, _, h, w = s.shape
            M = (torch.rand(b, 1, h, w, device=s.device) + self.update_prob).floor()

        k1 = self.adaptation(s, dx, dy)
        k2 = self.adaptation(s + k1 * 0.5 * M, dx, dy)
        k3 = self.adaptation(s + k2 * 0.5 * M, dx, dy)
        k4 = self.adaptation(s + k3 * M, dx, dy)
        return s + (k1 + 2 * k2 + 2 * k3 + k4) * dt * M / 6.0

    def forward(self, s, dx=1.0, dy=1.0, dt=1.0, integrator='euler'):
        """
        :param s: Cell states tensor of shape [b, chn, h, w]
        :param dx: Either a float or a tensor of shape [b, h, w]
        :param dy: Either a float or a tensor of shape [b, h, w]
        :param dt: Time step used for integration. Must be a float value <= 1.0
        :param integrator: Integration method. Either 'euler' or 'rk4'

        :return s: Updated cell states tensor of shape [b, chn, h, w]
        """
        if integrator == 'euler':
            return self.step_euler(s, dx, dy, dt)
        elif integrator == 'rk4':
            return self.step_rk4(s, dx, dy, dt)
        else:
            raise ValueError("Invalid integrator. Must be either 'euler' or 'rk4'")

    def seed(self, n, h=128, w=128):
        return (torch.rand(n, self.chn, h, w, device=self.device) - 0.5) * self.noise_level

    def to_rgb(self, s):
        return s[..., :3, :, :] + 0.5

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.device = self.w1.weight.device
        return self


if __name__ == "__main__":
    device = torch.device("cpu")
    # with torch.no_grad():
    nca = NCA(12, 96, device=device, noise_level=1.0)
    seed = nca.seed(1)
    s = nca(seed)
