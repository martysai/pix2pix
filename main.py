import numpy as np
import os
import random
from src.arguments import create_args
from models.gan import GAN
from models.init_weights import init_weights
import torch

from torch.utils.data import DataLoader
from dataset.dataset import MyDataset
from src.train import train, test
import subprocess
# import wandb


def count_parameters(model):
    gen_parameters = filter(lambda p: p.requires_grad,
                            model.generator.parameters())
    gen_num = sum([np.prod(p.size()) for p in gen_parameters])

    disc_parameters = filter(lambda p: p.requires_grad,
                             model.discriminator.parameters())
    disc_num = sum([np.prod(p.size()) for p in disc_parameters])
    return gen_num, disc_num


def fix_random_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


PREFIX_LINK = "http://efrosgans.eecs.berkeley.edu/pix2pix/datasets"
EDGES2SHOES_LINK = f"{PREFIX_LINK}/edges2shoes.tar.gz"
FACADES_LINK = f"{PREFIX_LINK}/facades.tar.gz"
NIGHT2DAY_LINK = f"{PREFIX_LINK}/night2day.tar.gz"

DATASETS_NAME2LINK = {
    "edges2shoes": EDGES2SHOES_LINK,
    "facades": FACADES_LINK,
    "night2day": NIGHT2DAY_LINK,
}


def get_filename(link):
    return link[link.rfind('/') + 1:]


def download_data(args):
    if os.path.exists(args.dataset):
        print(f"Dataset {args.dataset} has been loaded yet!")
        return

    # Если решили скачать всё, то уберите if с continue
    for dataset, link in DATASETS_NAME2LINK.items():
        if dataset != args.dataset:
            continue
        if args.verbose:
            print(f"Downloading {dataset}...")
        return_code = subprocess.run(["wget", link],
                                     stdout=subprocess.DEVNULL,
                                     stderr=subprocess.STDOUT)
        return_code = return_code.returncode
        assert return_code == 0, f"wget error during downloading {dataset}."

        filename = get_filename(link)
        return_code = subprocess.run(["tar", "-xzf", filename],
                                     stdout=subprocess.DEVNULL,
                                     stderr=subprocess.STDOUT)
        return_code = return_code.returncode
        assert return_code == 0, f"tar error during unpacking of {dataset}."
    if args.verbose:
        print("Finished datasets download.")
    return


def main():
    fix_random_seed(1024)
    np.set_printoptions(precision=2)

    args = create_args()
    if args.verbose:
        print(args)

    if not os.path.exists(args.test_dir):
        os.mkdir(args.test_dir)

    # Будем логировать, только если в режиме обучения
    # if args.training:
    #     wandb.login()
    #     wandb.init(project="Deep Learning Hometask 3")

    if args.download_data:
        download_data(args)

    model = GAN(args)
    model = model.to(args.device)

    if args.from_checkpoint or args.testing:
        # Загрузим с чекпоинта модель, два оптимайзера и два скедулера.
        checkpoint = torch.load(f"logs/checkpoints/best_checkpoint_{args.dataset}.pth")
        model.generator.load_state_dict(checkpoint["generator"])
        model.discriminator.load_state_dict(checkpoint["discriminator"])
        model.generator_opt.load_state_dict(checkpoint["generator_opt"])
        model.discriminator_opt.load_state_dict(
            checkpoint["discriminator_opt"])
        model.generator_scheduler.load_state_dict(
            checkpoint["generator_scheduler"])
        model.discriminator_scheduler.load_state_dict(
            checkpoint["discriminator_scheduler"])
        print("Loaded from checkpoint")
    else:
        # Если обучаем с нуля, инициализируем веса
        init_weights(model, is_kaiming=args.is_kaiming)

    if args.verbose:
        print("Model evaluation")
        print(model.eval())

        gen_num, disc_num = count_parameters(model)
        print("Total number of trainable parameters.")
        print("Generator:", gen_num)
        print("Discriminator:", disc_num)

    left_to_right = args.dataset != "facades"

    if args.training:
        train_dataset = MyDataset(args,
                                  data=args.dataset,
                                  mode="train",
                                  left_to_right=left_to_right)
        train_loader = DataLoader(train_dataset,
                                  batch_size=args.train_batch_size,
                                  shuffle=args.shuffle,
                                  num_workers=args.num_workers)
        valid_dataset = MyDataset(args,
                                  data=args.dataset,
                                  mode="val",
                                  left_to_right=left_to_right)
        valid_loader = DataLoader(valid_dataset,
                                  batch_size=args.valid_batch_size,
                                  shuffle=args.shuffle,
                                  num_workers=args.num_workers)

        if args.verbose:
            print("Train dataset size:", len(train_loader))
            print("Validation dataset size:", len(valid_loader))

        train(model, train_loader, valid_loader, args)

    if args.testing:
        test_dataset = MyDataset(args,
                                 data=args.dataset,
                                 mode="test",
                                 left_to_right=left_to_right)
        test_loader = DataLoader(test_dataset,
                                 batch_size=args.valid_batch_size,
                                 shuffle=args.shuffle,
                                 num_workers=args.num_workers)
        test(model, test_loader, 10, args)
    return


if __name__ == "__main__":
    main()
