import argparse
import struct
import torch

from model import EDSR

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--pth_path",
        type=str,
        default="",
        help="path of .pth",
    )
    parser.add_argument(
        "-o",
        "--wts_path",
        type=str,
        default="",
        help="path of .wts",
    )
    parser.add_argument(
        "-b",
        "--num_blocks",
        type=int,
        default=16,
        help="number of blocks that are used in the model",
    )
    parser.add_argument(
        "-n",
        "--num_feats",
        type=int,
        default=64,
        help="number of features that are used in the model",
    )
    parser.add_argument(
        "-r",
        "--res_scale",
        type=float,
        default=1.0,
        help="number of res-scale that are used in the model",
    )
    parser.add_argument(
        "-s",
        "--scale",
        type=int,
        default=2,
        help="number of scale that are used in the model",
    )
    parser.add_argument(
        "-c",
        "--channels",
        type=int,
        default=3,
        help="number of channels that are used in the model",
    )
    args = parser.parse_args()

    edsr = EDSR(
        args.scale,
        args.channels,
        args.channels,
        args.num_feats,
        args.num_blocks,
        args.res_scale,
    )

    ckpt = torch.load(args.pth_path, map_location=lambda storage, loc: storage)
    edsr.load_state_dict(ckpt["g"])

    if args.pth_path.endswith(".wts"):
        print("Already, .wts file exists.")
    elif args.pth_path.endswith(".pth"):
        print("Start to make EDSR.wts ...")
        f = open(args.wts_path, "w")
        f.write("{}\n".format(len(edsr.state_dict().keys())))
        for k, v in edsr.state_dict().items():
            print("key: ", k)
            print("value: ", v.shape)
            vr = v.reshape(-1).cpu().numpy()
            f.write("{} {}".format(k, len(vr)))
            for vv in vr:
                f.write(" ")
                f.write(struct.pack(">f", float(vv)).hex())
            f.write("\n")
        print(f"Completed to make .wts in {args.wts_path}")
    else:
        raise ValueError(f"{args.pth_path} is not proper model")
