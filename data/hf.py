from datasets import load_dataset, get_dataset_split_names
import os
import torchaudio
import torch

gs = load_dataset(
    "speechcolab/gigaspeech", "xs", use_auth_token=True, trust_remote_code=True
)

for i, item in enumerate(gs["train"]):
    audio_id = item["audio_id"]
    print(audio_id)
    output_dir = "datasets/audios"
    os.makedirs(output_dir, exist_ok=True)

    audio_filename = os.path.join(output_dir, audio_id) + ".wav"
    # print(audio_filename)
    audio_tensor = item["audio"]["array"]
    x = torch.FloatTensor(audio_tensor)
    x = x.unsqueeze(0)
    sample_rate = item["audio"]["sampling_rate"]
    torchaudio.save(uri=audio_filename, src=x, sample_rate=sample_rate, format="wav")
    # break
