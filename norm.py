# import torchaudio
# from speechbrain.inference.enhancement import WaveformEnhancement

# enhance_model = WaveformEnhancement.from_hparams(
#     source="speechbrain/mtl-mimic-voicebank",
#     savedir="pretrained_models/mtl-mimic-voicebank",
# )
# enhanced = enhance_model.enhance_file("speechbrain/mtl-mimic-voicebank/example.wav")

# # Saving enhanced signal on disk
# torchaudio.save('enhanced.wav', enhanced.unsqueeze(0).cpu(), 16000)

import requests

API_URL = "https://api-inference.huggingface.co/models/speechbrain/mtl-mimic-voicebank"
headers = {"Authorization": "Bearer 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"}

def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()

output = query("sample1.flac")