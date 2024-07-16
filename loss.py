import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import torchvision.models as torch_models


class TextureLoss(torch.nn.Module):
    def __init__(self, loss_type, **kwargs):
        """
        :param loss_type: 'OT', 'SlW', 'Gram'
        OT: Relaxed Optimal Transport Style loss proposed by Kolkin et al. https://arxiv.org/abs/1904.12785
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
        self.vgg = torch_models.vgg16(weights='IMAGENET1K_V1').features.to(self.device)
        self._create_losses()

    def _create_losses(self):
        self.loss_mapper = {}
        self.loss_weights = {}
        if self.slw_weight != 0:
            self.loss_mapper["SlW"] = SlicedWassersteinLoss(self.vgg)
            self.loss_weights["SlW"] = self.slw_weight

        if self.ot_weight != 0:
            self.loss_mapper["OT"] = RelaxedOTLoss(self.vgg)
            self.loss_weights["OT"] = self.ot_weight

        if self.gram_weight != 0:
            self.loss_mapper["Gram"] = GramLoss(self.vgg)
            self.loss_weights["Gram"] = self.gram_weight

    def forward(self, target_images, generated_images):
        loss = 0.0
        b, c, h, w = generated_images.shape
        _, _, ht, wt = target_images.shape
        if h != ht or w != wt:
            target_images = TF.resize(target_images, size=[h, w])
        for loss_name in self.loss_mapper:
            loss_weight = self.loss_weights[loss_name]
            loss_func = self.loss_mapper[loss_name]
            loss_per_image = loss_func(target_images, generated_images)
            loss += loss_weight * sum(loss_per_image) / len(loss_per_image)

        return loss, loss_per_image


class GramLoss(torch.nn.Module):
    def __init__(self, vgg):
        super(GramLoss, self).__init__()
        self.vgg = vgg

    @staticmethod
    def get_gram(y):
        b, c, h, w = y.size()
        features = y.view(b, c, w * h)
        features_t = features.transpose(1, 2)
        grams = features.bmm(features_t) / (h * w)
        return grams

    def forward(self, target_images, generated_images):
        with torch.no_grad():
            target_features = get_middle_feature_vgg(target_images, self.vgg)
        generated_features = get_middle_feature_vgg(generated_images, self.vgg)

        losses = []
        for target_feature, generated_feature in zip(target_features, generated_features):
            gram_target = self.get_gram(target_feature)
            gram_generated = self.get_gram(generated_feature)
            losses.append((gram_target - gram_generated).square().mean())
        return losses


class SlicedWassersteinLoss(torch.nn.Module):
    def __init__(self, vgg):
        super(SlicedWassersteinLoss, self).__init__()
        self.vgg = vgg

    @staticmethod
    def project_sort(x, proj):
        return torch.einsum('bcn,cp->bpn', x, proj).sort()[0]

    @staticmethod
    def sliced_wass_loss(source, target, proj_n=32):
        ch, n = source.shape[-2:]
        projs = F.normalize(torch.randn(ch, proj_n, device=source.device), dim=0)
        source_proj = SlicedWassersteinLoss.project_sort(source, projs)
        target_proj = SlicedWassersteinLoss.project_sort(target, projs)
        target_interp = F.interpolate(target_proj, n, mode='nearest')
        return (source_proj - target_interp).square().sum()

    def forward(self, target_images, generated_images, mask=None):
        with torch.no_grad():
            target_features = get_middle_feature_vgg(target_images, self.vgg, flatten=True,
                                                     include_image_as_feat=True)
        generated_features = get_middle_feature_vgg(generated_images, self.vgg, flatten=True,
                                                    include_image_as_feat=True)

        losses = [self.sliced_wass_loss(x, y) for x, y in zip(generated_features, target_features)]
        return losses


class RelaxedOTLoss(torch.nn.Module):
    """https://arxiv.org/abs/1904.12785"""

    def __init__(self, vgg, n_samples=1024):
        super().__init__()
        self.n_samples = n_samples
        self.vgg = vgg

    @staticmethod
    def pairwise_distances_cos(x, y):
        x_norm = torch.norm(x, dim=2, keepdim=True)  # (b, n, 1)
        y_t = y.transpose(1, 2)  # (b, c, m) (m may be different from n)
        y_norm = torch.norm(y_t, dim=1, keepdim=True)  # (b, 1, m)
        dist = 1. - torch.matmul(x, y_t) / (x_norm * y_norm + 1e-10)  # (b, n, m)
        return dist

    @staticmethod
    def style_loss(x, y):
        pairwise_distance = RelaxedOTLoss.pairwise_distances_cos(x, y)
        m1, m1_inds = pairwise_distance.min(1)
        m2, m2_inds = pairwise_distance.min(2)
        remd = torch.max(m1.mean(dim=1), m2.mean(dim=1))
        return remd

    @staticmethod
    def moment_loss(x, y):
        mu_x, mu_y = torch.mean(x, 1, keepdim=True), torch.mean(y, 1, keepdim=True)
        mu_diff = torch.abs(mu_x - mu_y).mean(dim=(1, 2))

        x_c, y_c = x - mu_x, y - mu_y
        x_cov = torch.matmul(x_c.transpose(1, 2), x_c) / (x.shape[1] - 1)
        y_cov = torch.matmul(y_c.transpose(1, 2), y_c) / (y.shape[1] - 1)

        cov_diff = torch.abs(x_cov - y_cov).mean(dim=(1, 2))
        return mu_diff + cov_diff

    def forward(self, target_images, generated_images):
        loss = 0.0
        with torch.no_grad():
            target_features = get_middle_feature_vgg(target_images, self.vgg, flatten=True)
        generated_features = get_middle_feature_vgg(generated_images, self.vgg, flatten=True)
        # Iterate over the VGG layers
        for x, y in zip(generated_features, target_features):
            (b_x, c, n_x), (b_y, _, n_y) = x.shape, y.shape
            n_samples = min(n_x, n_y, self.n_samples)
            indices_x = torch.argsort(torch.rand(b_x, 1, n_x, device=x.device), dim=-1)[..., :n_samples]
            x = x.gather(-1, indices_x.expand(b_x, c, n_samples))
            indices_y = torch.argsort(torch.rand(b_y, 1, n_y, device=y.device), dim=-1)[..., :n_samples]
            y = y.gather(-1, indices_y.expand(b_y, c, n_samples))
            x, y = x.transpose(1, 2), y.transpose(1, 2)  # (b, n_samples, c)
            loss += self.style_loss(x, y) + self.moment_loss(x, y)

        return loss


def get_middle_feature_vgg(imgs, vgg_model, flatten=False, include_image_as_feat=False):
    style_layers = [1, 6, 11, 18, 25]  # 1, 6, 11, 18, 25
    mean = torch.tensor([0.485, 0.456, 0.406], device=imgs.device)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=imgs.device)[:, None, None]
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
