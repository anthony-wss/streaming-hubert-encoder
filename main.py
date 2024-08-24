import os
from argparse import ArgumentParser
from streaming_hubert import StreamingHubertEncoder, ApplyKmeans


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--audio_dir", required=True, help="Path to audio folder")
    parser.add_argument("--ext", default="wav", help="Audio extention name")
    parser.add_argument("--km_model", default="./km_model-1s.pt", help="Path to the Kmeans model")
    args = parser.parse_args()

    file_list = []
    prefix = args.audio_dir
    prefix = args.audio_dir
    for file in os.listdir(prefix):
        if not file.endswith(args.ext):
            continue
        file_list.append(os.path.join(prefix, file))


    # Step 1: Get causal hubert hidden feature at lajyer 6
    encoder = StreamingHubertEncoder()
    feats = encoder.batch_encode(file_list)
    print([f.shape for f in feats])

    # Step 2: Kmeans quantization
    apply_kmeans = ApplyKmeans(args.km_model, use_gpu=True)
    ssl_units = [apply_kmeans(feat) for feat in feats]
    print([len(seq) for seq in ssl_units])

