# Streaming HuBert Encoder

- [x] feat: support multi-gpu feature extraction
- [x] feat: support infinite window size
- [x] feat: Add warning if `output_dir` is not empty
- [x] refactor: Move `HOP_LENGTH`, `WIN_LENGTH`, and `batch_size` to `StreamingHubertEncoder` config
- [x] feat: Support not dumping features with `dump_feature` parameter
- [ ] test: Add unit test `test.py`
- [x] feat: `StreamingHubertEncoder` only takes list of path strings as input
- [x] fix : Fix `UserWarning: To copy construct from a tensor`
- [x] feat: Support mean downsampling

## Install

```bash
git clone https://github.com/anthony-wss/streaming-hubert-encoder
cd streaming-hubert-encoder
pip install -r requirements
pip install .
```

## Usage

The audio in `audio_dir` should be in wav format.
`window_sec` can be 1, 5 or -1.

```bash
python main.py \
    --audio_dir /work/u3937558/soundon-asr/audio_10 \
    --output_dir test_data_dump \
    --window_sec -1 \
    --take_mean \
    --km_model ../discrete-chinese-hubert-base-l6/km_500_inf.pt
```

## Usage (Multi-gpu)

1. Set parameters in `multi_gpu_extractor.sh`

2. Run

```bash
bash multi_gpu_extractor.sh
```

## Document

`multi_gpu_extractor.sh` will do the following things:

1. Split `$audio_dir` into N shards for N gpus.
2. Run inference for each gpu. Log would be in `$log_dir/gpu_i.log`

Note: 
- Press `Ctrl+C` will terminate all processes.
- All the audio file in `$audio_dir` should be in `.wav` format, 16k sampling rate.

