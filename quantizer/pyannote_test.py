from pyannote.audio import Inference
from pyannote.audio import Pipeline

# pipeline = Pipeline.from_pretrained(
#     "pyannote/speaker-diarization-3.1",
#     use_auth_token="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")

spkr_embedding = Inference(
    "pyannote/embedding",
    window="whole",
)
