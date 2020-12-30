import torch
import torch.nn as nn
from torch.optim import Adam
from .generator import Generator
from .discriminator import Discriminator
from .models_helpers import create_scheduler, form_target_tensor


class GAN(nn.Module):
    def __init__(self, args):
        super(GAN, self).__init__()
        self.args = args

        # Определим компоненты GAN
        self.generator = Generator(args.dataset)

        self.discriminator = Discriminator(n_layers=args.n_layers)
        self.generator.to(args.device)
        self.discriminator.to(args.device)

        # Определим оптимайзеры
        self.generator_opt = Adam(self.generator.parameters(),
                                  betas=(0.5, 0.999),
                                  lr=args.lr)
        self.discriminator_opt = Adam(self.discriminator.parameters(),
                                      betas=(0.5, 0.999),
                                      lr=args.lr)

        # Определим скедулеры
        self.generator_scheduler = \
            create_scheduler(self.generator_opt, args)
        self.discriminator_scheduler = \
            create_scheduler(self.discriminator_opt, args)

        # Определим функции потерь
        self.gan_loss = nn.BCEWithLogitsLoss()
        self.l1_loss = nn.L1Loss()

    def set_grads(self, submodule, grads=False):
        for param in submodule.parameters():
            param.requires_grad = grads
        return

    def scheduler_step(self):
        self.generator_scheduler.step()
        self.discriminator_scheduler.step()
        return

    def forward(self, src):
        """
        При вызове модели применяем трансфер модальностей,
        т.е. генерируем ненастоящее изображение.
        """
        self.fake_trg = self.generator(src)
        return

    def optimize(self, src, trg):
        """
        Делаем по шагу оптимизации для дискриминатора и генератора.
        """
        self.forward(src)

        self.discriminator_step(src, trg)
        self.generator_step(src, trg)
        return

    def discriminator_step(self, src, trg):
        """
        Определим шаг дискриминатора.
        """
        self.set_grads(self.discriminator, True)
        self.discriminator_opt.zero_grad()

        # Сохраним фейковую пару для дискриминатора
        fake_pair = torch.cat((src, self.fake_trg), 1)
        discriminator_pred = self.discriminator(fake_pair.detach())

        target = form_target_tensor(discriminator_pred.size(),
                                    self.args.device,
                                    which="fake")
        self.fake_disc_loss = self.gan_loss(discriminator_pred, target)

        real_pair = torch.cat((src, trg), 1)
        discriminator_pred = self.discriminator(real_pair)

        target = form_target_tensor(discriminator_pred.size(),
                                    self.args.device,
                                    which="real")
        self.real_disc_loss = self.gan_loss(discriminator_pred, target)

        self.disc_loss = (1.0 / 2.0) * (self.fake_disc_loss +
                                        self.real_disc_loss)
        self.disc_loss.backward()

        self.discriminator_opt.step()
        return

    def generator_step(self, src, trg):
        """
        Определим шаг генератора.
        """
        self.set_grads(self.discriminator, False)
        self.generator_opt.zero_grad()

        # Сохраним фейковую пару для генератора
        fake_pair = torch.cat((src, self.fake_trg), 1)
        discriminator_pred = self.discriminator(fake_pair)

        target = form_target_tensor(discriminator_pred.size(),
                                    self.args.device,
                                    which="real")
        self.real_gen_loss = self.gan_loss(discriminator_pred, target)

        self.l1_gen_loss = self.args.coef * self.l1_loss(self.fake_trg, trg)
        self.gen_loss = self.real_gen_loss + self.l1_gen_loss
        self.gen_loss.backward()

        self.generator_opt.step()
        return
