import numpy as np
import os
import torch

PTH_PREFIX = "logs/checkpoints"


def mv_to_numpy(tensor):
    return tensor[0, :, :, :].permute(1, 2, 0).cpu().detach().numpy()


# Будем объединять изображения перед логированием
def unite_images(src, pred, trg):
    # Объединим по горизонтали
    src = mv_to_numpy(src)
    pred = mv_to_numpy(torch.clamp(pred, 0., 1.))
    trg = mv_to_numpy(trg)

    united = np.concatenate((src, pred, trg), 1)
    return united


def save_network(save_name, model, epoch, args):
    torch.save(
        {
            "generator": model.generator.state_dict(),
            "discriminator": model.discriminator.state_dict(),
            "generator_opt": model.generator_opt.state_dict(),
            "discriminator_opt": model.discriminator_opt.state_dict(),
            "discriminator_scheduler":
            model.discriminator_scheduler.state_dict(),
            "generator_scheduler": model.generator_scheduler.state_dict(),
            "epoch": epoch,
        }, os.path.join(PTH_PREFIX, save_name))
    return


# Посчитаем KL-дивергенцию по распределениям цветов
epsilon = 1e-12
colors = {0: "red", 1: "green", 2: "blue"}


# Этой функцией я считал KL-дивергенцию для facades
def count_kl_divergence(q, p):
    global epsilon
    return np.sum(q * np.log(q + epsilon)) - \
        np.sum(q * np.log(p + epsilon))


def retrieve_color_distributions(pred, trg, args):
    KL_values = {}
    for i, color in colors.items():
        # if args.verbose:
        #     print(f"Handling color = {color}")
        pred_dist = pred[:, i, :, :].detach().cpu().numpy().reshape(-1)
        pred_dist = (pred_dist + 1.0) / 2.0  # go to [0, 1]
        trg_dist = trg[:, i, :, :].detach().cpu().numpy().reshape(-1)
        KL_value = count_kl_divergence(pred_dist, trg_dist)
        KL_values[color] = KL_value
    return KL_values


# Поняв, что в PyTorch она реализована, стал делать нативным образом
# (для edges2shoes)
kl_loss = torch.nn.KLDivLoss(reduction='sum')


def retrieve_color_distributions_native(pred, trg, args):
    global kl_loss
    KL_values = {}
    for i, color in colors.items():
        KL_value = kl_loss(pred[:, i, :, :], trg[:, i, :, :])
        KL_values[color] = KL_value
    return KL_values


# Продублируем функцию из models_helpers.py, чтобы
# импортировать в train
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
