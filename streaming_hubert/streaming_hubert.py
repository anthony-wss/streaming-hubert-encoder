import torch.nn.functional as F
import soundfile as sf
from transformers import (
    Wav2Vec2FeatureExtractor,
    HubertModel,
)
from tqdm import tqdm
import torch
import numpy as np
import librosa
import os


class StreamingHubertEncoder():
    def __init__(self, output_dir, window_sec, hop_ms, batch_size=16, device="cuda"):
        model_path = "TencentGameMate/chinese-hubert-base"
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.device = device
        self.window_size = -1 if window_sec == -1 else window_sec * 16000
        self.hop_length = hop_ms * 16
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
        self.model = HubertModel.from_pretrained(model_path)
        self.model = self.model.to(self.device)
        self.model.eval()


    def batch_encode(self, audio_list):
        """
        Encode a list of audio

        Parameters:
            audio_list(list(str)): list of audio path strings

        Returns:
            feat: a list of Hubert representations for each file
        """
        feats = []
        shard_id = 0
        for i in tqdm(range(len(audio_list))):
            audio_id = audio_list[i].strip().split("/")[-1].split(".")[0]
            wav, sr = sf.read(audio_list[i])
            if sr != 16000:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
            
            wav_feat = []
            if self.window_size == -1:
                chunk_size = 16000*300
                for i in range(0, wav.shape[0], chunk_size):
                    end_pos = min(i+chunk_size, wav.shape[0])
                    batch_feats, batch_lens = self._encode([wav[i:end_pos]])
                    wav_feat.extend(batch_feats[0, :, :])
            else:
                wav_slices = []
                for i in range(self.hop_length, wav.shape[0], self.hop_length):
                    start_pos = max(i-self.window_size, 0)
                    wav_slices.append(wav[start_pos:i])

                    if len(wav_slices) >= self.batch_size or i+self.hop_length >= wav.shape[0]:
                        batch_feats, batch_lens = self._encode(wav_slices)
                        for bi in range(len(batch_feats)):
                            wav_feat.extend(batch_feats[bi][:batch_lens[bi]][-5:])
                            # print([l.shape for l  in wav_feat])
                            # exit()
                        wav_slices = []
                        torch.cuda.empty_cache()

            wav_feat = torch.vstack(wav_feat)
            # print(wav_feat.shape)
            # feats.append(wav_feat)

            file_path = os.path.join(self.output_dir, f"{audio_id}.pt")
            torch.save({
                "feats": wav_feat
            }, file_path)
            shard_id += 1

        return []


    def encode(self, audio_input):
        feats = self.batch_encode([audio_input])
        return feats[0]


    def _encode(self, wavs):
        """
        Encode list of audio into Hubert features

        Parameters:
            wavs: list of np.ndarray

        Returns:
            feats: list of torch.tensor, representing the L6 Hubert features
            lens: list of integer, representing the lengths of the features
        """
        is_batch = (len(wavs) > 1)
        wavs = [torch.from_numpy(wav) for wav in wavs]
        max_len = max(wav.shape[0] for wav in wavs)
        wavs_padded = [F.pad(wav, (0, max_len - wav.shape[0])) for wav in wavs]
        wavs_padded = torch.vstack(wavs_padded).squeeze()

        input_values = self.feature_extractor(wavs_padded, return_tensors="pt", sampling_rate=16000).input_values
        input_values = input_values.to(self.device)
        if is_batch:
            input_values = input_values.squeeze()
        outputs = self.model(input_values, attention_mask=torch.ones(input_values.shape[0]).to(self.device), output_hidden_states=True)
        feats = outputs.hidden_states[6].detach().cpu()
        lens = [(l.shape[0]-80)//320 for l in wavs]

        return feats, lens

