import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

#TRIEU-ADDED
class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        vgg = torchvision.models.vgg19(pretrained=True).features[:36].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.criterion = nn.L1Loss()

    def forward(self, x, y):
        x_vgg = self.vgg(x)
        y_vgg = self.vgg(y)
        return self.criterion(x_vgg, y_vgg)
class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        self.criterion = nn.L1Loss()

    def forward(self, x, y):
        x_edges = self._compute_edges(x)
        y_edges = self._compute_edges(y)
        return self.criterion(x_edges, y_edges)

    def _compute_edges(self, img):
        # Convert to grayscale if input is RGB
        if img.shape[1] == 3:
            img = 0.2989 * img[:, 0:1, :, :] + 0.5870 * img[:, 1:2, :, :] + 0.1140 * img[:, 2:3, :, :]

        sobel_x = torch.tensor([[[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]]], dtype=torch.float32, device=img.device).unsqueeze(0)  # [1, 1, 3, 3]
        sobel_y = torch.tensor([[[-1, -2, -1],
                                [ 0,  0,  0],
                                [ 1,  2,  1]]], dtype=torch.float32, device=img.device).unsqueeze(0)

        grad_x = F.conv2d(img, sobel_x, padding=1)
        grad_y = F.conv2d(img, sobel_y, padding=1)

        edge = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
        return edge


def generator_loss(fake_img, real_img):
    pixel = self.lambda_pixel * self.mse_loss(fake_img, real_img)
    perceptual = self.lambda_vgg * self.vgg_loss(fake_img, real_img)
    edge = self.lambda_edge * self.edge_loss(fake_img, real_img)
    return pixel + perceptual + edge


######

# Define GAN loss: [vanilla | lsgan | wgan-gp]
class GANLoss(nn.Module):
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        self.mse_loss = nn.MSELoss()
        self.vgg_loss = VGGLoss()  # Custom class using pretrained VGG features
        self.edge_loss = EdgeLoss()  # Optional, for better structure
        self.lambda_pixel = 1.0
        self.lambda_vgg = 1.0 #0.1
        self.lambda_edge = 1.0 #0.05

        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan-gp':
            def wgan_loss(input, target):
                # target is boolean
                return -1 * input.mean() if target else input.mean()
            self.loss = wgan_loss
        elif self.gan_type=='tmt':
            def generator_loss(fake_img, real_img):
                if fake_img.shape[1] == 1:
                    fake_img = fake_img.repeat(1, 3, 1, 1)  # Convert 1-channel to 3-channel
                if real_img.shape[1] == 1:
                    real_img = real_img.repeat(1, 3, 1, 1)

                fake_img = F.interpolate(fake_img, size=(128, 128), mode='bilinear', align_corners=False)
                real_img = F.interpolate(real_img, size=(128, 128), mode='bilinear', align_corners=False)


                pixel = self.lambda_pixel * self.mse_loss(fake_img, real_img)
                perceptual = self.lambda_vgg * self.vgg_loss(fake_img, real_img)
                edge = self.lambda_edge * self.edge_loss(fake_img, real_img)
                return pixel + perceptual + edge
            self.loss = generator_loss
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))

    def get_target_label(self, input, target_is_real):
        if self.gan_type == 'wgan-gp':
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss


class GradientPenaltyLoss(nn.Module):
    def __init__(self, device=torch.device('cpu')):
        super(GradientPenaltyLoss, self).__init__()
        self.register_buffer('grad_outputs', torch.Tensor())
        self.grad_outputs = self.grad_outputs.to(device)

    def get_grad_outputs(self, input):
        if self.grad_outputs.size() != input.size():
            self.grad_outputs.resize_(input.size()).fill_(1.0)
        return self.grad_outputs

    def forward(self, interp, interp_crit):
        grad_outputs = self.get_grad_outputs(interp_crit)
        grad_interp = torch.autograd.grad(outputs=interp_crit, inputs=interp, \
            grad_outputs=grad_outputs, create_graph=True, retain_graph=True, only_inputs=True)[0]
        grad_interp = grad_interp.view(grad_interp.size(0), -1)
        grad_interp_norm = grad_interp.norm(2, dim=1)

        loss = ((grad_interp_norm - 1)**2).mean()
        return loss
