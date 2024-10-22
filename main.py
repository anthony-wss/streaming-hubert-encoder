import os
from argparse import ArgumentParser
from streaming_hubert import StreamingHubertEncoder, ApplyKmeans
from streaming_hubert.utils import cal_duplicate_tokens


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--audio_dir", default=None, help="Path to audio folder")
    parser.add_argument("--file_list", default=None, help="A text file with audio paths. One for each line.")
    parser.add_argument("--output_dir", required=True, help="Directory to store Hubert features")
    parser.add_argument("--ext", default="wav", help="Audio extention name")
    parser.add_argument("--km_model", default="./km_model-1s.pt", help="Path to the Kmeans model")
    parser.add_argument("--window_sec", type=int, default=1, help="Window size in second, set -1 for inf")
    parser.add_argument("--hop_ms", type=int, default=100, help="Hop length in millisecond")
    parser.add_argument("--batch_size", type=int, default=100, help="Number of windows per batch")
    parser.add_argument("--take_mean", action="store_true", help="Use mean downsampling")
    parser.add_argument("--dump_feature", action="store_true", help="Dump the feature to output_dir")
    args = parser.parse_args()

    if args.audio_dir is not None:
        file_list = []
        for file in os.listdir(args.audio_dir):
            if not file.endswith(args.ext):
                continue
            file_list.append(os.path.join(args.audio_dir, file))
    elif args.file_list is not None:
        file_list = [l.strip() for l in open(args.file_list, "r").readlines()]
    else:
        raise Exception("you should set audio_dir or file_list")

    if args.dump_feature and len(os.listdir(args.output_dir)) > 0:
        print(f"Warning: the output_dir {args.output_dir} is not empty")


    encoder = StreamingHubertEncoder(
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        window_sec=args.window_sec,
        hop_ms=args.hop_ms,
        take_mean=args.take_mean,
        dump_feature=args.dump_feature
    )

    # Step 1: Get causal hubert hidden feature at layer 6
    feats = encoder.batch_encode(file_list)

    # Step 2: Kmeans quantization
    apply_kmeans = ApplyKmeans(args.km_model, use_gpu=True)
    ssl_units = [apply_kmeans(feat) for feat in feats]

    mean, std = cal_duplicate_tokens(ssl_units)
    print(mean, std)
