# Streaming HuBert Encoder (Multi-gpu branch)

- [x] feat: support multi-gpu feature extraction
- [ ] feat: support infinite window size
- [ ] feat: Add warning if `output_dir` is not empty
- [x] refactor: Move `HOP_LENGTH`, `WIN_LENGTH`, and `batch_size` to `StreamingHubertEncoder` config
- [ ] feat: Support not dumping features with `dump_feature` parameter
- [ ] test: Add unit test `test.py`
- [x] feat: `StreamingHubertEncoder` only takes list of path strings as input
- [ ] fix : ```/work/u3937558/streaming-hubert/streaming_hubert/streaming_hubert.py:98: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).```

## Install

```bash
git clone https://github.com/anthony-wss/streaming-hubert-encoder
cd streaming-hubert-encoder
pip install -r requirements
pip install .
```

## Usage


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

