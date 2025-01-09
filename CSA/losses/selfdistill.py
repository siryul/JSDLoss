import torch
import torch.nn.functional as F


class JSDDistillLoss(torch.nn.Module):

  def __init__(self, cls_num_list, temperature=1.0, lam_min=0.01, lam_max=0.1):
    super(JSDDistillLoss, self).__init__()
    cls_num_list = torch.cuda.FloatTensor(cls_num_list)
    self.weights = cls_num_list.max() / cls_num_list
    self.temperature = temperature
    self.lam_min = lam_min
    self.lam_max = lam_max

  def forward(self, logits_a, logits_b, targets, epoch, epochs):
    pa_softmax = F.softmax(logits_a / self.temperature, dim=1)
    pb_softmax = F.softmax(logits_b / self.temperature, dim=1)

    epsilon = 1e-10
    kl_div_a = pa_softmax * (torch.log(pa_softmax + epsilon) - torch.log(pb_softmax + epsilon))
    kl_div_b = pb_softmax * (torch.log(pb_softmax + epsilon) - torch.log(pa_softmax + epsilon))
    kl_div = (kl_div_a.sum(dim=1) + kl_div_b.sum(dim=1)) / 2

    class_weights = self.weights[targets]

    jsd = torch.mean(kl_div * class_weights)

    lam = self.lam_min + (self.lam_max - self.lam_min) * epoch / epochs

    return F.cross_entropy(logits_a, targets) + F.cross_entropy(logits_b, targets) + lam*jsd


def JSDLoss(logits_a,
            logits_b,
            targets,
            weights,
            epoch,
            epochs,
            temperature=1.0,
            lam_min=0.01,
            lam_max=0.1):
  pa_softmax = F.softmax(logits_a / temperature, dim=1)
  pb_softmax = F.softmax(logits_b / temperature, dim=1)

  epsilon = 1e-10
  kl_div_a = pa_softmax * (torch.log(pa_softmax + epsilon) - torch.log(pb_softmax + epsilon))
  kl_div_b = pb_softmax * (torch.log(pb_softmax + epsilon) - torch.log(pa_softmax + epsilon))
  kl_div = (kl_div_a.sum(dim=1) + kl_div_b.sum(dim=1)) / 2

  class_weights = weights[targets]

  jsd = torch.mean(kl_div * class_weights)

  lam = lam_min + (lam_max-lam_min) * epoch / epochs

  return F.cross_entropy(logits_a, targets) + F.cross_entropy(logits_b, targets) + lam*jsd
