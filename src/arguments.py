import argparse
import torch


def create_args():
    parser = argparse.ArgumentParser(
        'Deep Learning Hometask 3',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Базовые аргументы
    basic = parser.add_argument_group('Basic')
    basic.add_argument('--epoch_count', type=int, default=1)
    basic.add_argument('--n_epochs', type=int, default=500)
    basic.add_argument('--n_epochs_decay', type=int, default=50)
    basic.add_argument('--verbose',
                       type=bool,
                       default=True,
                       help='additional verbosity')
    basic.add_argument('--download_data', action='store_false')
    basic.add_argument('--train_batch_size', type=int, default=4)
    basic.add_argument('--valid_batch_size', type=int, default=1)
    basic.add_argument('--logging_dir', type=str, default="logs")
    basic.add_argument('--pred_dir', type=str, default="logs/pred")
    basic.add_argument('--test_dir', type=str, default="logs/test")

    # Число слоев в дискриминаторе
    basic.add_argument('--n_layers', type=int, default=4)

    # Тип инициализации
    basic.add_argument('--is_kaiming',
                       type=bool,
                       default=False,
                       help="choose initalization with kaiming or xavier")

    # Частоты
    freqs = parser.add_argument_group('Frequencies')
    freqs.add_argument('--print_freq', type=int, default=30)
    freqs.add_argument('--save_epoch_freq', type=int, default=1)

    # Оптимизатор
    opts = parser.add_argument_group('Optimizer')
    opts.add_argument('--lr', type=float, default=0.0002)
    opts.add_argument('--coef', type=float, default=100.0)

    # Помощники
    helpers = parser.add_argument_group('Helpers')
    helpers.add_argument('--preprocess', type=str, default='resize_and_crop')
    helpers.add_argument('--load_size', type=int, default=286)
    helpers.add_argument('--crop_size', type=int, default=256)

    # Датасет
    dataset = parser.add_argument_group('Dataset')
    dataset.add_argument('--shuffle',
                         type=bool,
                         default=False,
                         help="shuffle in data loader or not")
    dataset.add_argument('--num_workers', default=8, type=int)

    # Флаги для обучения / тестирования
    modes = parser.add_argument_group('Mode')
    modes.add_argument('--training', type=bool, default=True)
    modes.add_argument('--dataset', type=str, default="facades")
    modes.add_argument('--testing', type=bool, default=False)
    modes.add_argument('--from_checkpoint', type=bool, default=False)
    modes.add_argument('--no_validation', type=bool, default=False)

    args, unknown = parser.parse_known_args()
    args.cuda = torch.cuda.is_available()
    args.device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.
                               cuda.is_available() else "cpu")

    return args
