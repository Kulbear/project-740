from torch.optim import lr_scheduler

__all__ = [
    'StepLR',
    'CyclicLR',
    'ReduceLROnPlateau'
]


def StepLR(optimizer,
           metadata={}):
    gamma = metadata.get('gamma', 0.1)
    step_size = metadata.get('step_size', 30)
    last_epoch = metadata.get('last_epoch', -1)
    return lr_scheduler.StepLR(optimizer,
                               step_size,
                               gamma=gamma,
                               last_epoch=last_epoch)


def ReduceLROnPlateau(optimizer,
                      metadata={}):
    eps = metadata.get('eps', 1e-08)
    mode = metadata.get('mode', 'min')
    min_lr = metadata.get('min_lr', 0)
    factor = metadata.get('factor', 0.1)
    cooldown = metadata.get('cooldown', 0)
    patience = metadata.get('patience', 10)
    verbose = metadata.get('verbose', False)
    threshold = metadata.get('threshold', 0.0001)
    threshold_mode = metadata.get('threshold_mode', 'rel')

    return lr_scheduler.ReduceLROnPlateau(optimizer,
                                          eps=eps,
                                          mode=mode,
                                          min_lr=min_lr,
                                          factor=factor,
                                          verbose=verbose,
                                          cooldown=cooldown,
                                          patience=patience,
                                          threshold=threshold,
                                          threshold_mode=threshold_mode
                                          )


def CyclicLR(optimizer,
             metadata={}):
    gamma = metadata.get('gamma', 0.98)
    max_lr = metadata.get('max_lr', 1e-2)
    base_lr = metadata.get('base_lr', 1e-6)
    mode = metadata.get('mode', 'exp_range')
    scale_fn = metadata.get('scale_fn', None)
    last_epoch = metadata.get('last_epoch', -1)
    scale_mode = metadata.get('scale_mode', 'cycle')
    max_momentum = metadata.get('max_momentum', 0.9)
    base_momentum = metadata.get('base_momentum', 0.8)
    step_size_down = metadata.get('step_size_down', 2000)
    cycle_momentum = metadata.get('cycle_momentum', True)

    return lr_scheduler.CyclicLR(optimizer,
                                 base_lr,
                                 max_lr,
                                 mode=mode,
                                 gamma=gamma,
                                 scale_fn=scale_fn,
                                 last_epoch=last_epoch,
                                 scale_mode=scale_mode,
                                 max_momentum=max_momentum,
                                 base_momentum=base_momentum,
                                 step_size_down=step_size_down,
                                 cycle_momentum=cycle_momentum
                                 )
