import numpy as np
import os
import torch
import time
# import wandb
# Используем KL-дивергенцию как часть количественной оценки
from .main_helpers import save_network, retrieve_color_distributions, \
                          unite_images, retrieve_color_distributions_native, \
                          form_target_tensor
from PIL import Image

# ----
# Логирование я закомментировал, чтобы было видно, что оно было.
# ----


def count_grad_norm(parameters, norm_type=2.0):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    total_norm = torch.norm(
        torch.stack([
            torch.norm(p.grad.detach(), norm_type).to(device)
            for p in parameters
        ]), norm_type)
    return total_norm.item()


def mv_to_numpy(tensor):
    return tensor[0, :, :, :].permute(1, 2, 0).cpu().detach().numpy()


def train_epoch(model, train_loader, epoch, args):
    model.train()

    epoch_gen_loss = 0.
    epoch_disc_loss = 0.

    for i, train_batch in enumerate(train_loader):
        src = train_batch["src"]
        trg = train_batch["trg"]
        src, trg = src.to(args.device), trg.to(args.device)

        model.optimize(src, trg)

        epoch_gen_loss += model.gen_loss.item()
        epoch_disc_loss += model.disc_loss.item()

        if i % args.print_freq == 0:
            # Посчитаем град. норму для контроля взрыва/затухания
            grad_norm = count_grad_norm(model.parameters())
            print(f"[{i}/{len(train_loader)}] grad_norm = ",
                  f"{grad_norm:.2f}")
            print(f"[{i}/{len(train_loader)}] epoch_gen_loss = "
                  f"{epoch_gen_loss / (i + 1):.2f}")
            print(f"[{i}/{len(train_loader)}] epoch_disc_loss = "
                  f"{epoch_disc_loss / (i + 1):.2f}")

            united = unite_images(src, model.fake_trg, trg)
            save_pred_name = os.path.join(args.pred_dir,
                                          f"train-e{epoch}-i{i}.png")
            img = Image.fromarray((255.0 * united).astype(np.uint8))
            img.save(save_pred_name)

            # if epoch % 5 == 0 or args.dataset != 'facades':
            #     wandb.log({
            #         "[Step/Train] Generator loss":
            #         model.gen_loss.item(),
            #         "[Step/Train] Discriminator loss":
            #         model.disc_loss.item(),
            #         "[Step/Train] Gradient norm":
            #         grad_norm,
            #         "Train/source image": [wandb.Image(mv_to_numpy(src))],
            #         "Train/predicted image":
            #         [wandb.Image(mv_to_numpy(model.fake_trg))],
            #         "Train/target image": [wandb.Image(mv_to_numpy(trg))],
            #     })

        torch.cuda.empty_cache()

    epoch_gen_loss /= len(train_loader)
    epoch_disc_loss /= len(train_loader)
    return epoch_gen_loss, epoch_disc_loss


def train(model, train_loader, valid_loader, args):
    best_valid_loss = np.inf
    epoch_start_time = time.time()

    # Для фасадов выполняем валидацию реже
    evaluate_frequency = 1
    if args.dataset == 'facades':
        evaluate_frequency = 3

    for epoch in range(args.epoch_count, args.n_epochs + 1):
        train_gen_loss, train_disc_loss = \
            train_epoch(model, train_loader, epoch, args)

        print(f"Epoch {epoch:.2f} / {args.n_epochs:.2f}:")
        print(f"Elapsed: {time.time() - epoch_start_time:.2f} sec.")
        print(f"Train generator loss: {train_gen_loss:.2f}")
        print(f"Train discriminator loss: {train_disc_loss:.2f}")

        if epoch % evaluate_frequency == 0:
            valid_gen_loss, valid_disc_loss, \
                valid_kl, valid_l1_loss = \
                evaluate(model, valid_loader, epoch, args)

            mean_kl = (valid_kl["red"] + valid_kl["blue"] +
                       valid_kl["green"]) / 3.

            print(f"Valid mean KL-divergence: {mean_kl:.2f}")
            print(f"Valid L1 Loss: {valid_l1_loss:.2f}")
            print(f"Valid KL for red, green, blue: "
                  f"{valid_kl['red']:.2f}, "
                  f"{valid_kl['green']:.2f}, "
                  f"{valid_kl['blue']:.2f}")

            print(f"Valid generator loss: {valid_gen_loss:.2f}")
            print(f"Valid discriminator loss: {valid_disc_loss:.2f}")
            # wandb.log({
            #     "[Epoch/Train] Generator loss": train_gen_loss,
            #     "[Epoch/Train] Discriminator loss": train_disc_loss,
            #     "[Epoch/Valid] Generator loss": valid_gen_loss,
            #     "[Epoch/Valid] Discriminator loss": valid_disc_loss,
            #     "[Epoch/Valid] KL divergence": mean_kl,
            #     "[Epoch/Valid] L1 loss": valid_l1_loss,
            # })

            if valid_gen_loss < best_valid_loss:
                best_valid_loss = valid_gen_loss
                print(f"Best valid generator loss: {best_valid_loss:.2f}")
                name = "best_checkpoint.pth"
                save_network(name, model, epoch, args)

            if epoch % 4 == 0:
                name = f"checkpoint-e{epoch}.pth"
                save_network(name, model, epoch, args)

            model.scheduler_step()

    return


def evaluate(model, valid_loader, epoch, args):
    # Убираем .eval()-режим, должны быть более "живые"
    # предсказания
    # model.eval()
    valid_gen_loss, valid_disc_loss = 0., 0.
    kl_divergence = {
        "red": 0.,
        "green": 0.,
        "blue": 0.,
    }
    # Будем логировать значение L_1 как отдельную часть количеств.
    # оценки
    valid_l1_loss = 0.

    with torch.no_grad():
        for i, valid_batch in enumerate(valid_loader):
            src = valid_batch["src"]
            trg = valid_batch["trg"]
            src, trg = src.to(args.device), trg.to(args.device)

            model(src)
            fake_pair = torch.cat((src, model.fake_trg), 1)
            discriminator_pred = model.discriminator(fake_pair)

            # Сформируем целевые тензоры
            fake_target = form_target_tensor(discriminator_pred.size(),
                                             args.device,
                                             which="fake")
            real_target = form_target_tensor(discriminator_pred.size(),
                                             args.device,
                                             which="real")

            fake_disc_loss = model.gan_loss(discriminator_pred, fake_target)
            real_gen_loss = model.gan_loss(discriminator_pred, real_target)

            l1_gen_loss = args.coef * model.l1_loss(model.fake_trg, trg)
            gen_loss = real_gen_loss + l1_gen_loss

            real_pair = torch.cat((src, trg), 1)
            discriminator_pred = model.discriminator(real_pair)
            real_disc_loss = model.gan_loss(discriminator_pred, real_target)

            disc_loss = (fake_disc_loss + real_disc_loss) / 2.0

            valid_gen_loss += gen_loss.item()
            valid_disc_loss += disc_loss.item()
            valid_l1_loss += l1_gen_loss.item()

            # Посчитаем KL-дивергенцию для данного батча
            if args.dataset == "facades":
                kl_result = retrieve_color_distributions(
                    model.fake_trg, trg, args)
            else:
                kl_result = retrieve_color_distributions_native(
                    model.fake_trg, trg, args)
            for color, value in kl_result.items():
                kl_divergence[color] += value

            united = unite_images(src, model.fake_trg, trg)
            save_pred_name = os.path.join(args.pred_dir,
                                          f"valid-e{epoch}-i{i}.png")
            img = Image.fromarray((255.0 * united).astype(np.uint8))
            img.save(save_pred_name)

            # Логирование происходит каждую 3-ю эпоху
            # wandb.log({
            #     "Valid/source image": [wandb.Image(mv_to_numpy(src))],
            #     "Valid/predicted image":
            #     [wandb.Image(mv_to_numpy(model.fake_trg))],
            #     "Valid/target image": [wandb.Image(mv_to_numpy(trg))],
            # })

    for color in kl_divergence.keys():
        kl_divergence[color] /= len(valid_loader)

    return valid_gen_loss / len(valid_loader), \
        valid_disc_loss / len(valid_loader), \
        kl_divergence, valid_l1_loss / len(valid_loader)


def test(model, test_loader, epoch_count, args):
    # Убираем .eval()-режим, должны быть более "живые"
    # предсказания
    # model.eval()
    with torch.no_grad():
        for i, test_batch in enumerate(test_loader):
            src = test_batch["src"]
            trg = test_batch["trg"]
            src, trg = src.to(args.device), trg.to(args.device)
            model(src)

            united = unite_images(src, model.fake_trg, trg)
            save_pred_name = os.path.join(args.test_dir,
                                          f"test-e{epoch_count}-i{i}.png")
            img = Image.fromarray((255.0 * united).astype(np.uint8))
            img.save(save_pred_name)
    return
