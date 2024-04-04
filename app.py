# import requests
# import io
# import re
# from PIL import Image
# from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
# from datasets import load_dataset
# import torch
# import soundfile as sf
# from datasets import load_dataset
# API_URL = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
# headers = {"Authorization":"Bearer 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"}
# def query(payload):
# 	response = requests.post(API_URL, headers=headers, json=payload)
# 	return response.content
# '''
# image_bytes = query({
# 	"inputs": "Astronaut riding a horse",
# })
# '''

# def break_story_into_frames(story):
#   sentences=story.split('.')
#   prompts=[]
#   all=[]
#   for sentence in sentences:
#     prompt=sentence.split(',')
#     prompts.append(prompt)
#   for prompt in prompts:
#     for i in range(len(prompt)):
#       all.append(prompt[i])
#   return all
# '''
# all=break_story_into_frames('hello guys.How are you')
# Diction=[]
# for frame in all:
#     Diction.append({'inputs':frame})
# '''    
# import random
# def Story(story):
#     frames = break_story_into_frames(story)
#     images = []

#     for frame in frames:
#         image=query(frame)
#         images.append(image)
#     return images
# image=Story('hello guys.How are you')
# '''
# img = Image.open(io.BytesIO(image[0]))
# img.show()
# '''
# #print(len(image))
# processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
# model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
# vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
# def decoder(frames):
#   waveforms=[]
#   for frame in frames:
#     inputs=processor(text=frame, return_tensors="pt")
#     speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
#     speech_waveform = speech.numpy()
#     waveforms.append(speech_waveform)

#   return waveforms
# embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
# speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
# story='riya is pretty.she is 18'
# frames=break_story_into_frames(story)
# waveforms=decoder(frames)
# print(len(image))
# print(len(waveforms))

# for i in range(len(image)):
#     display_image_and_play_speech(images[i], waveforms[i])

import requests
import io
from PIL import Image
import soundfile as sf
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import os
import platform

# Define your API URL and headers
API_URL = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
headers = {"Authorization": "Bearer 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"}

# Function to query the API for image
def query_image(text):
    response = requests.post(API_URL, headers=headers, json={"inputs": text})
    return response.content

# Function to display image and play speech
def display_image_and_play_speech(image_bytes, speech_waveform):
    # Display image
    img = Image.open(io.BytesIO(image_bytes))
    img.show()

    # Play speech
    sf.write('output.wav', speech_waveform, 22050)
    play_audio("output.wav")

# Function to play audio based on OS
def play_audio(audio_file):
    if platform.system() == "Windows":
        os.system("start " + audio_file)
    elif platform.system() == "Darwin":  # macOS
        os.system("afplay " + audio_file)
    else:  # Linux
        os.system("aplay " + audio_file)

# Load speech-to-text models
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# Load speaker embeddings
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

# Function to decode frames into speech
def decode_frames_to_speech(frames):
    waveforms = []
    for frame in frames:
        inputs = processor(text=frame, return_tensors="pt")
        speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
        speech_waveform = speech.numpy()
        waveforms.append(speech_waveform)
    return waveforms

# Sample story
story = 'Modiji Mila Russian president Putin se.Modiji ne bhasan diya wahan.Modiji Plane Pakda aur India Wapas aagaya'

# Break story into frames
frames = story.split('.')

# Get image for each frame
images = [query_image(frame) for frame in frames]

# Decode frames into speech
waveforms = decode_frames_to_speech(frames)

# Display images and play speech
for i in range(len(images)):
    display_image_and_play_speech(images[i], waveforms[i])
