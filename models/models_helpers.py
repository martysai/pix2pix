import torch


def create_scheduler(optimizer, args):
    def lr_lambda(epoch):
        nominator = max(0, epoch + args.epoch_count - args.n_epochs)
        denominator = float(args.n_epochs_decay + 1)
        inverse_lr = nominator / denominator
        return 1.0 - inverse_lr

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                  lr_lambda=lr_lambda)
    return scheduler


def form_target_tensor(size, device, which="fake"):
    """
    Тензор, используемый для подсчета BCEWithLogitsLoss
    """
    if which == "real":
        target = torch.full(size, 1.0).to(device)
    elif which == "fake":
        target = torch.full(size, 0.0).to(device)
    else:
        raise ValueError("Неизвестное значение prediction_type.")
    return target
