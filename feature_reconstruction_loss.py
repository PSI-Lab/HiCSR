import torch
from torch import nn
from models.DAE_model import DAE

class FeatureReconstructionLoss(nn.Module):
    def __init__(self, layers = [0,1,2,3,4]):
        super(FeatureReconstructionLoss, self).__init__()
        loss_net = DAE()
        loss_net.load_state_dict(torch.load('./pretrained_models/DAE.pth'))
        encoder = list(loss_net.children())[0]

        self.sub_nets = []
        self.layers = layers

        for layer in layers:
            list_of_layers = list(encoder)[:layer]
            final_layer = [encoder[layer][0]]
            sub_net = nn.Sequential(*(list_of_layers + final_layer)).float().eval().cuda()

            for param in sub_net.parameters():
                param.requires_grad = False

            self.sub_nets.append(sub_net)

        self.loss = nn.MSELoss()

    def forward(self, prediction, target):

        feature_loss = torch.tensor([0.0]).float().cuda()

        for net in self.sub_nets:
            pred_feat = net(prediction)
            target_feat = net(target)
            loss = self.loss(pred_feat, target_feat)
            feature_loss += loss

        return feature_loss


