# Streaming HuBert Encoder

## Install

```bash
git clone https://github.com/anthony-wss/streaming-hubert-encoder
cd streaming-hubert-encoder
pip install -r requirements
pip install .
```

## Usage

1. Transform all the audio in a folder into Hubert unit

```bash
python main.py --audio_dir [path] --ext wav
```

2. Transform one audio into hubert unit

```python
file_path = "./test.flac"
enc = StreamingHubertEncoder()
feat, leng = enc.encode(file_path)
```

