import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import torchvision.models as torch_models


class TextureLoss(torch.nn.Module):
    def __init__(self, loss_type, **kwargs):
        """
        :param loss_type: 'OT', 'SlW', 'Gram'
        OT: Optimal Transport Style loss proposed by Kolkin et al. https://arxiv.org/abs/1904.12785
        SlW: Sliced Wasserstein Style loss proposed by Heitz et al. https://arxiv.org/abs/2006.07229
        Gram: Gram Style loss proposed by Gatys et al. in https://arxiv.org/abs/1505.07376
        """
        super(TextureLoss, self).__init__()

        self.ot_weight = 0.
        self.slw_weight = 0.
        self.gram_weight = 0.

        self.loss_type = loss_type

        if loss_type == 'OT':
            self.ot_weight = 1.0
        elif loss_type == 'SlW':
            self.slw_weight = 1.0
        elif loss_type == 'Gram':
            self.gram_weight = 1.0

        self.device = kwargs['device']
        self._create_losses()

    def _create_losses(self):
        self.loss_mapper = {}
        self.loss_weights = {}
        if self.slw_weight != 0:
            self.loss_mapper["SlW"] = SlicedWassersteinLoss(self.device)
            self.loss_weights["SlW"] = self.slw_weight

        if self.ot_weight != 0:
            self.loss_mapper["OT"] = OptimalTransportLoss(self.device)
            self.loss_weights["OT"] = self.ot_weight

        if self.gram_weight != 0:
            self.loss_mapper["Gram"] = GramLoss(self.device)
            self.loss_weights["Gram"] = self.gram_weight

    def forward(self, target_images, generated_images, mask=None):
        loss = 0.0
        b, c, h, w = generated_images.shape
        _, _, ht, wt = target_images.shape
        if h != ht or w != wt:
            target_images = TF.resize(target_images, size=(h, w))
        for loss_name in self.loss_mapper:
            loss_weight = self.loss_weights[loss_name]
            loss_func = self.loss_mapper[loss_name]
            losses = loss_func(target_images, generated_images, mask)
            loss += loss_weight * sum(losses) / len(losses)

        return loss, losses


class GramLoss(torch.nn.Module):
    def __init__(self, device):
        super(GramLoss, self).__init__()
        self.device = device
        self.vgg16 = torch_models.vgg16(weights=torch_models.VGG16_Weights.IMAGENET1K_V1).features.to(device)

    @staticmethod
    def get_gram(y):
        b, c, h, w = y.size()
        features = y.view(b, c, w * h)
        features_t = features.transpose(1, 2)
        grams = features.bmm(features_t) / (h * w)
        return grams

    def forward(self, target_images, generated_images, mask=None):
        with torch.no_grad():
            target_features = get_middle_feature_vgg(target_images, self.vgg16, device=self.device)
        generated_features = get_middle_feature_vgg(generated_images, self.vgg16, device=self.device)

        losses = []
        for target_feature, generated_feature in zip(target_features, generated_features):
            gram_target = self.get_gram(target_feature)
            gram_generated = self.get_gram(generated_feature)
            losses.append((gram_target - gram_generated).square().mean())
        return losses


class SlicedWassersteinLoss(torch.nn.Module):
    def __init__(self, device):
        super(SlicedWassersteinLoss, self).__init__()
        self.device = device
        self.vgg16 = torch_models.vgg16(weights=torch_models.VGG16_Weights.IMAGENET1K_V1).features.to(device)

    @staticmethod
    def project_sort(x, proj):
        return torch.einsum('bcn,cp->bpn', x, proj).sort()[0]

    def sliced_ot_loss(self, source, target, proj_n=32):
        ch, n = source.shape[-2:]
        projs = F.normalize(torch.randn(ch, proj_n, device=self.device), dim=0)
        source_proj = SlicedWassersteinLoss.project_sort(source, projs)
        target_proj = SlicedWassersteinLoss.project_sort(target, projs)
        target_interp = F.interpolate(target_proj, n, mode='nearest')
        return (source_proj - target_interp).square().sum()

    def forward(self, target_images, generated_images, mask=None):
        with torch.no_grad():
            target_features = get_middle_feature_vgg(target_images, self.vgg16, flatten=True,
                                                     include_image_as_feat=True, device=self.device)
        generated_features = get_middle_feature_vgg(generated_images, self.vgg16, flatten=True,
                                                    include_image_as_feat=True, device=self.device)

        losses = [self.sliced_ot_loss(x, y) for x, y in zip(generated_features, target_features)]
        return losses


class OptimalTransportLoss(torch.nn.Module):
    def __init__(self, device):
        super(OptimalTransportLoss, self).__init__()
        self.device = device
        self.vgg16 = torch_models.vgg16(weights=torch_models.VGG16_Weights.IMAGENET1K_V1).features.to(device)

    @staticmethod
    def pairwise_distances_cos(x, y):
        x_norm = torch.sqrt((x ** 2).sum(1).view(-1, 1))
        y_t = torch.transpose(y, 0, 1)
        y_norm = torch.sqrt((y ** 2).sum(1).view(1, -1))
        dist = 1. - torch.mm(x, y_t) / (x_norm + 1e-10) / (y_norm + 1e-10)
        return dist

    @staticmethod
    def style_loss_cos(X, Y, cos_d=True):
        # X,Y: 1*d*N*1
        d = X.shape[1]

        X = X.transpose(0, 1).contiguous().view(d, -1).transpose(0, 1)  # N*d
        Y = Y.transpose(0, 1).contiguous().view(d, -1).transpose(0, 1)

        # Relaxed EMD
        CX_M = OptimalTransportLoss.pairwise_distances_cos(X, Y)

        m1, m1_inds = CX_M.min(1)
        m2, m2_inds = CX_M.min(0)

        remd = torch.max(m1.mean(), m2.mean())

        return remd

    @staticmethod
    def moment_loss(X, Y):  # matching mean and cov
        X = X.squeeze().t()
        Y = Y.squeeze().t()

        mu_x = torch.mean(X, 0, keepdim=True)
        mu_y = torch.mean(Y, 0, keepdim=True)
        mu_d = torch.abs(mu_x - mu_y).mean()

        X_c = X - mu_x
        Y_c = Y - mu_y
        X_cov = torch.mm(X_c.t(), X_c) / (X.shape[0] - 1)
        Y_cov = torch.mm(Y_c.t(), Y_c) / (Y.shape[0] - 1)

        D_cov = torch.abs(X_cov - Y_cov).mean()
        loss = mu_d + D_cov

        return loss

    def perm_gpu_f32(self, pop_size, num_samples):
        """Use torch.randperm to generate indices on a 32-bit GPU tensor."""
        return torch.randperm(pop_size, dtype=torch.long, device=self.device)[:num_samples]

    def get_ot_loss_single_batch(self, x_feature, y_feature):
        randomize = True
        loss = 0
        for x, y in zip(x_feature, y_feature):
            c = x.shape[1]
            h, w = x.shape[2], x.shape[3]
            x = x.reshape(1, c, -1, 1)
            y = y.reshape(1, c, -1, 1)
            if h > 32 and randomize:
                indices = self.perm_gpu_f32(h * w, 1000)
                # indices = np.random.choice(np.arange(h * w), size=1000, replace=False)
                # indices = np.sort(indices)
                # indices = torch.LongTensor(indices)
                x = x[:, :, indices, :]
                y = y[:, :, indices, :]
            loss += OptimalTransportLoss.style_loss_cos(x, y)
            loss += OptimalTransportLoss.moment_loss(x, y)
        return loss

    def forward(self, target_images, generated_images, mask=None):
        with torch.no_grad():
            target_features = get_middle_feature_vgg(target_images, self.vgg16, device=self.device)
        generated_features = get_middle_feature_vgg(generated_images, self.vgg16, device=self.device)

        batch_size = target_images.shape[0]
        losses = []
        for b in range(batch_size):
            target_feature = [t[b:b + 1] for t in target_features]
            generated_feature = [g[b:b + 1] for g in generated_features]
            losses.append(self.get_ot_loss_single_batch(target_feature, generated_feature))
        return losses


def get_middle_feature_vgg(imgs, vgg_model, flatten=False, include_image_as_feat=False, device="cuda:0"):
    style_layers = [1, 6, 11, 18, 25]  # 1, 6, 11, 18, 25
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[:, None, None]
    x = (imgs - mean) / std
    b, c, h, w = x.shape
    if include_image_as_feat:
        features = [x.reshape(b, c, h * w)]
    else:
        features = []
    for i, layer in enumerate(vgg_model[:max(style_layers) + 1]):
        x = layer(x)
        if i in style_layers:
            b, c, h, w = x.shape
            if flatten:
                features.append(x.reshape(b, c, h * w))
            else:
                features.append(x)
    return features
