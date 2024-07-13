import torch


def depthwise_conv(x, filters, padding='circular'):
    """filters: [filter_n, h, w]"""
    b, ch, h, w = x.shape
    y = x.reshape(b * ch, 1, h, w)
    y = torch.nn.functional.pad(y, [1, 1, 1, 1], padding)
    y = torch.nn.functional.conv2d(y, filters[:, None])
    return y.reshape(b, -1, h, w)


def merge_lap(z):
    # This function merges the lap_x and lap_y into a single laplacian filter
    b, c, h, w = z.shape
    z = torch.stack([
        z[:, ::5],
        z[:, 1::5],
        z[:, 2::5],
        z[:, 3::5] + z[:, 4::5]
    ],
        dim=2)  # [b, chn, 4, h, w]
    return z.reshape(b, -1, h, w)  # [b, 4 * chn, h, w]


class NCA(torch.nn.Module):
    """
    Base class for Neural Cellular Automata.
    The functionalities to change the scale of perception filters and to add conditional channels are included in
    the base class. The extensions such as NoiseNCA and PENCA are implemented by inheriting from this class.
    """

    def __init__(self, chn, fc_dim,
                 padding='circular', perception_kernels=4,
                 cond_chn=0, update_prob=0.5, device=None):
        """
        chn: Number of channels in the cell state
        fc_dim: Number of channels in the update MLP hidden layer
        padding: Padding mode for the perception (Convolution kernels)
        perception_kernels: Number of perception kernels. The baseline NCA uses 4 kernels: Identity, Sobel X, Sobel Y, Laplacian
        cond_chn: Number of conditional channels.
                  For example a 2D positional encoding will add 2 extra condition channels. If the number of conditional
                  channels is > 0 then you need to override the adaptation/perception methods and provide the extra condition channels.
        update_prob: Probability of updating a cell state in each iteration.
                     If update_prob = 1.0, all the cells are updated in each iteration.
        device: PyTorch device
        """
        super(NCA, self).__init__()
        self.chn, self.fc_dim, self.padding, self.perception_kernels = chn, fc_dim, padding, perception_kernels
        self.cond_chn, self.update_prob, self.device = cond_chn, update_prob, device

        self.w1 = torch.nn.Conv2d(chn * perception_kernels + cond_chn, fc_dim, 1, bias=True, device=device)
        self.w2 = torch.nn.Conv2d(fc_dim, chn, 1, bias=False, device=device)

        torch.nn.init.xavier_normal_(self.w1.weight, gain=0.2)
        torch.nn.init.zeros_(self.w2.weight)

        with torch.no_grad():
            ident = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]], device=device)
            sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=device)
            lap_x = torch.tensor([[0.5, 0.0, 0.5], [2.0, -6.0, 2.0], [0.5, 0.0, 0.5]], device=device)

            self.filters = torch.stack([ident, sobel_x, sobel_x.T, lap_x, lap_x.T])

    def perception(self, s, dx=1.0, dy=1.0):
        """
        Computes the perception vector for each cell given the current cell states s.
        dx, dy are used to scale the sobel and laplacian filters.
        dx, dy < 1.0 means that the patterns are gonna get stretched horizontally, vertically.
        dx, dy > 1.0 means that the patterns are gonna get squeezed horizontally, vertically.
        """

        z = depthwise_conv(s, self.filters, self.padding)  # [b, 5 * chn, h, w]
        if dx == 1.0 and dy == 1.0:
            return merge_lap(z)

        if not isinstance(dx, torch.Tensor) or dx.ndim != 3:
            dx = torch.tensor([dx], device=s.device)[:, None, None]  # [1, 1, 1]
        if not isinstance(dy, torch.Tensor) or dy.ndim != 3:
            dy = torch.tensor([dy], device=s.device)[:, None, None]  # [1, 1, 1]

        scale = 1.0 / torch.stack([torch.ones_like(dx), dx, dy, dx ** 2, dy ** 2], dim=1)
        scale = torch.tile(scale, (1, self.chn, 1, 1))
        z = z * scale
        return merge_lap(z)

    def adaptation(self, s, dx=1.0, dy=1.0):
        """Computes the residual update given current cell states s"""
        z = self.perception(s, dx, dy)
        delta_s = self.w2(torch.relu(self.w1(z)))
        return delta_s

    def step_euler(self, s, dx=1.0, dy=1.0, dt=1.0):
        """Computes one step of the NCA update using the Euler integrator."""
        delta_s = self.adaptation(s, dx, dy)
        M = 1.0
        if self.update_prob < 1.0:
            b, _, h, w = s.shape
            M = (torch.rand(b, 1, h, w, device=s.device) + self.update_prob).floor()

        return s + delta_s * M * dt

    def step_rk4(self, s, dx=1.0, dy=1.0, dt=1.0):
        """Computes one step of the NCA update using the 4th order Runge-Kutta integrator."""
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
        Computes one step of the NCA update rule using the specified integrator.

        :param s: Cell states tensor of shape [b, chn, h, w]
        :param dx: Either a float or a tensor of shape [b, h, w]
        :param dy: Either a float or a tensor of shape [b, h, w]
        :param dt: Time step used for integration. Must be a float value <= 1.0
        :param integrator: Integration method. Either 'euler' or 'rk4'
        """
        if integrator == 'euler':
            return self.step_euler(s, dx, dy, dt)
        elif integrator == 'rk4':
            return self.step_rk4(s, dx, dy, dt)
        else:
            raise ValueError("Invalid integrator. Must be either 'euler' or 'rk4'")

    def seed(self, n, h=128, w=128):
        """Starting cell state"""
        return torch.zeros(n, self.chn, h, w, device=self.device)

    def to_rgb(self, s):
        """Converts the cell state to RGB. The offset of 0.5 is added so that the valid values are in [0, 1] range."""
        return s[..., :3, :, :] + 0.5

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.filters = self.filters.to(*args, **kwargs)
        self.device = self.w1.weight.device
        return self


class NoiseNCA(NCA):
    """
    NoiseNCA model where the seed is initialized with uniform noise.
    The functionalities to change the scale of the patterns and rate of
     the pattern formation are implemented in the base NCA class.
    """

    def __init__(self, chn, fc_dim, noise_level=0.0, **kwargs):
        """
        noise_level: Noise level for the seed initialization. 0.0 means that the seed is initialized with zeros.
                 noise_level = 1.0 means that the seed is initialized with uniform noise in [-0.5, 0.5].
        """
        assert "update_prob" not in kwargs, "The update probability is fixed to 1.0 for NoiseNCA."
        super(NoiseNCA, self).__init__(chn, fc_dim, update_prob=1.0, **kwargs)
        self.register_buffer("noise_level", torch.tensor([noise_level], device=self.device))

    def seed(self, n, h=128, w=128):
        return (torch.rand(n, self.chn, h, w, device=self.device) - 0.5) * self.noise_level


class PENCA(NCA):
    """
    PENCA is a baseline NCA model with an additional 2D positional encoding as conditional channels.
    The architecture is a simplified version of DyNCA model https://arxiv.org/abs/2211.11417.
    """

    def __init__(self, chn, fc_dim, **kwargs):
        assert "cond_chn" not in kwargs, "The number of conditional channels is fixed to 2 for PENCA."
        super(PENCA, self).__init__(chn, fc_dim, cond_chn=2, **kwargs)

        self.cached_grid = None
        self.last_shape = None

    def adaptation(self, s, dx=1.0, dy=1.0):
        z = self.perception(s, dx, dy)

        if self.cached_grid is None and self.last_shape == s.shape:
            grid = self.cached_grid
        else:
            b, _, h, w = s.shape
            xs, ys = torch.arange(h, device=s.device) / h, torch.arange(w, device=s.device) / w
            xs, ys = 2.0 * (xs - 0.5 + 0.5 / h), 2.0 * (ys - 0.5 + 0.5 / w)
            xs, ys = xs[None, :, None], ys[None, None, :]
            grid = torch.zeros((2, h, w), device=s.device, dtype=s.dtype)
            grid[:1], grid[1: 2] = xs, ys
            grid = grid.unsqueeze(0).repeat(b, 1, 1, 1)  # [b, 2, h, w]
            self.last_shape = s.shape
            self.cached_grid = grid

        z = torch.cat([z, grid], dim=1)
        delta_s = self.w2(torch.relu(self.w1(z)))
        return delta_s


if __name__ == "__main__":
    device = torch.device("cuda")
    model = NoiseNCA(12, 96, noise_level=0.1, device=device)

    state_dict = torch.load("weights.pt", map_location="cpu")
    model.load_state_dict(state_dict)

    from tqdm import tqdm
    from utils.video_utils import VideoWriter
    with VideoWriter() as vid, torch.no_grad():
        s = model.seed(1, 512, 512).to(device)
        for k in tqdm(range(600)):
            step_n = 8
            for i in range(step_n):
                s[:] = model(s)

            img = model.to_rgb(s[0]).permute(1, 2, 0).cpu()
            vid.add(img)

