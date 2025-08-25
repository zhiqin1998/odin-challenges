# from .data_finetune import build_loader_finetune
from .data_pretrain import build_loader_simmim


def build_loader(args, is_pretrain):
    if is_pretrain:
        return build_loader_simmim(args)
    else:
        raise NotImplementedError
        return build_loader_finetune(args)
