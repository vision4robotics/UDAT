def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay*init_lr
        lr=param_group['lr']
    return lr


def adjust_lr_finetune(optimizer, base_lr, finetune_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        if param_group["name"] == "base_params":
            param_group['lr'] = decay*base_lr
            base_lr = param_group['lr']
        if param_group["name"] == "finetune_params":
            param_group['lr'] = decay*finetune_lr
            finetune_lr = param_group['lr']

    return base_lr, finetune_lr