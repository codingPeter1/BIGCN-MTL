import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


class his_Loss(nn.Module):
    def __init__(self):
        super(his_Loss, self).__init__()

    def forward(self, now, his):
        now = F.normalize(now, dim=-1)
        his = F.normalize(his, dim=-1)
        # logger.info(f"now: {now}")
        # logger.info(f"his: {his}")
        cos_sim = torch.sum(now * his, dim=-1)
        # logger.info(f"cos_sim: {cos_sim}")
        loss = -torch.log((cos_sim+1)/2)
        loss = loss.mean()
        return loss

class MFLogLoss(nn.Module):
    def __init__(self):
        super(MFLogLoss, self).__init__()
        self.gamma = 1

    def forward(self, scores, labels):
        loss = torch.log(1 + torch.exp(-scores * labels))
        loss = loss.mean()
        return loss

class BCR_loss(nn.Module):
    def __init__(self):
        super(BCR_loss, self).__init__()

    def forward(self, score1, score2, inner_weight):
        loss = (1-torch.sigmoid(inner_weight*(score1-score2)))**2
        loss = loss.mean()
        return loss

class BPRLoss(nn.Module):
    def __init__(self):
        super(BPRLoss, self).__init__()
        self.gamma = 1e-10

    def forward(self, p_score, n_score):
        loss = -torch.log(self.gamma + torch.sigmoid(p_score - n_score))
        loss = loss.mean()

        return loss

class EmbLoss(nn.Module):
    """ EmbLoss, regularization on embeddings

    """

    def __init__(self, norm=2):
        super(EmbLoss, self).__init__()
        self.norm = norm

    def forward(self, *embeddings):
        emb_loss = torch.zeros(1).to(embeddings[-1].device)
        for embedding in embeddings:
            emb_loss += torch.norm(embedding, p=self.norm)
        emb_loss /= embeddings[-1].shape[0]
        return emb_loss