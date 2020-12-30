from torch.nn import init


def init_weights(model, is_kaiming=True):
    def init_func(layer):
        layer_name = layer.__class__.__name__
        # Проверим, является ли слой конволюцией
        if hasattr(layer, 'weight') and (layer_name.find('Conv') != -1):
            if is_kaiming:
                init.kaiming_normal_(layer.weight.data, mode='fan_in')
            else:
                init.xavier_normal_(layer.weight.data, gain=0.02)
            if hasattr(layer, 'bias') and \
               layer.bias is not None:
                init.constant_(layer.bias.data, 0.0)
        # Проверим, является ли слой батчнормой
        elif layer_name.find('BatchNorm2d') != -1:
            init.normal_(layer.weight.data, 1.0, 0.02)
            init.constant_(layer.bias.data, 0.0)

    model.apply(init_func)
    return
