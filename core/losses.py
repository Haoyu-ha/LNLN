from torch import nn
from torch.nn import functional as F


class MultimodalLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.alpha = args['base']['alpha']
        self.beta = args['base']['beta']
        self.gamma = args['base']['gamma']
        self.sigma = args['base']['sigma']
        self.CE_Fn = nn.CrossEntropyLoss()
        self.MSE_Fn = nn.MSELoss() 


    def forward(self, out, label):

        l_cc = self.MSE_Fn(out['w'], label['completeness_labels']) if out['w'] is not None else 0

        l_adv = self.CE_Fn(out['effectiveness_discriminator_out'], label['effectiveness_labels']) if out['effectiveness_discriminator_out'] is not None else 0

        l_rec = self.MSE_Fn(out['rec_feats'], out['complete_feats']) if out['rec_feats'] is not None and out['complete_feats'] is not None else 0

        l_sp = self.MSE_Fn(out['sentiment_preds'], label['sentiment_labels'])
        
        loss = self.alpha * l_cc + self.beta * l_adv + self.gamma * l_rec + self.sigma * l_sp

        return {'loss': loss, 'l_sp': l_sp, 'l_cc': l_cc, \
                'l_adv': l_adv, 'l_rec': l_rec}

