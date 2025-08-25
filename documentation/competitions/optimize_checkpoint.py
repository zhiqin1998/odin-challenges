import os
import argparse
import pickle
import torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--clean_conf', action='store_true', default=True)
    # parser.add_argument('--aot_compile', action='store_true')
    args = parser.parse_args()

    assert args.input_path != args.out_path

    print('removing unnecessary stuff')
    ckpt = torch.load(args.input_path, map_location='cpu', weights_only=False)
    # remove unnecessary stuff
    ckpt = {k: ckpt[k] for k in ['network_weights', 'init_args', 'trainer_name', 'inference_allowed_mirroring_axes']}
    if args.fp16:
        for k, v in ckpt['network_weights'].items():
            ckpt['network_weights'][k] = v.half()
    if args.clean_conf:
        conf_name = ckpt['init_args']['configuration']
        to_del = list(ckpt['init_args']['plans']['configurations'].keys())
        curr_conf = ckpt['init_args']['plans']['configurations'][conf_name]
        to_del.remove(conf_name)
        while 'inherits_from' in curr_conf:
            parent = curr_conf['inherits_from']
            curr_conf = ckpt['init_args']['plans']['configurations'][parent]
            to_del.remove(parent)
        for k in to_del:
            print(f'removing {k} from plans')
            del ckpt['init_args']['plans']['configurations'][k]

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    torch.save(ckpt, args.out_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)
