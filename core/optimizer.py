import torch

def get_optimizer(model, opt):
    params_visual_backbone_id = list(map(id, model.visual_backbone.parameters()))
    params_audio_backbone_id = list(map(id, model.audio_backbone.parameters()))
    params_fusion = filter(lambda p: id(p) not in params_visual_backbone_id + params_audio_backbone_id,
                           model.parameters())
    optimizer = torch.optim.Adam([{'params': model.visual_backbone.parameters(), 'lr': 0.1 * opt.lr},
                                  {'params': model.audio_backbone.parameters(), 'lr': 0.1 * opt.lr},
                                  {'params': params_fusion, 'lr': opt.lr}],
                                 lr=opt.lr,
                                 weight_decay=opt.weight_decay)

    return optimizer