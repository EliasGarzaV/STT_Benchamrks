#%%
from evaluate import load
from pydub import AudioSegment
import os
import json
import requests
import pandas as pd
from dotenv import load_dotenv
import json
from deepgram import DeepgramClient
from elevenlabs.client import ElevenLabs
load_dotenv()

DEEPGRAMG_API_KEY = os.getenv("DEEPGRAM_API_KEY")
CARTESIA_API_KEY = os.getenv("CARTESIA_KEY")
ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY')

#%% Loading reviewed transcripts

file_path = "transcriptions.txt"

with open(file_path, "r", encoding="utf-8") as file:
    transcripts = [line.strip() for line in file.readlines()]
    
#%% Converting to mp3

def convert_opus_to_mp3(opus_file, mp3_file):
    opus_file_path = opus_file
    mp3_file_path = mp3_file
    
    audio = AudioSegment.from_file(opus_file_path)
    audio.export(mp3_file_path, format="mp3")
    
    return mp3_file_path

for i in range(len(transcripts)):
    convert_opus_to_mp3(f'audios\opus\OPUS_{i}.opus', f'audios\mp3\AudioBenchmark_{i}.mp3')


#%%Deepgram API wrapper

def stt_deepgram(audio_path:str, client:DeepgramClient, model_str:str='nova-3') -> str:
    try:
        with open(audio_path, "rb") as audio_file:
            response = client.listen.v1.media.transcribe_file(
                request=audio_file.read(),
                model=model_str,
                language='es',
                smart_format=True,
            )

        return json.loads(response.json())['results']['channels'][0]['alternatives'][0]['transcript']
    except Exception as e:
        raise e

deepgram_client = DeepgramClient(api_key=DEEPGRAMG_API_KEY)


# stt_deepgram('audios/mp3/AudioBenchmark_1.mp3', deepgram_client)

#%% Cartesia API wrapper

def stt_cartesia(audio_path: str) -> str:
    try:
        url = "https://api.cartesia.ai/stt"
        headers = {
            "Authorization": f"Bearer {CARTESIA_API_KEY}",
            "Cartesia-Version": '2025-04-16'
        }
        files = {"file": open(audio_path, "rb")}
        data = {
            "model": "ink-whisper",
            "language": "es",
            "timestamp_granularities[]": "word"
        }

        response = requests.post(url, headers=headers, files=files, data=data)
        response.raise_for_status()
        result = response.json()
        # results_Cartesia returns `text` in root or inside `transcript`
        return result.get("text") or result.get("transcript", "")
    except Exception as e:
        raise e

# stt_cartesia('audios/mp3/AudioBenchmark_1.mp3')

#%%Elevenlabs API Wrapper

def stt_elevenlabs(audio_path: str) -> str:
    try:
        # STEP 2: Call the transcribe_file method with the audio file and options
        with open(audio_path, "rb") as audio_file:
            transcription = elevenlabs.speech_to_text.convert(
                file=audio_file,
                model_id="scribe_v1", # Model to use, for now only "scribe_v1" is supported
                tag_audio_events=True, # Tag audio events like laughter, applause, etc.
                language_code="spa", # Language of the audio file. If set to None, the model will detect the language automatically.
                diarize=False, # Whether to annotate who is speaking
            )
            
            return transcription.dict()['text']
    except Exception as e:
        raise e

elevenlabs = ElevenLabs(api_key=ELEVENLABS_API_KEY)

# stt_elevenlabs('audios/mp3/AudioBenchmark_1.mp3')

#%% WER definition

def get_wer(predictions, real, wer_model=load("wer"), evaluate_punctuation:bool=True, evaluate_accent:bool=False, remove=".,?!¿¡;:*(){}[]'"):
    #All to lower case
    predictions = [p.lower() for p in predictions]
    real = [p.lower() for p in real]
    
    if not evaluate_accent:
        replacement_dict = {'á':'a', 'é':'e', 'í':'i', 'ó':'o', 'ú':'u'}
        predictions = [''.join([replacement_dict[char] if char in replacement_dict else char for char in p]) for p in predictions]
        real = [''.join([replacement_dict[char] if char in replacement_dict else char for char in r]) for r in real]
    
    if not evaluate_punctuation:
        # remove_set = set(".,?!¿¡;:*(){}[]'")
        remove_set = set(remove)
        predictions = [''.join(c for c in p if c not in remove_set) for p in predictions]
        real = [''.join(c for c in r if c not in remove_set) for r in real]
        
    return wer_model.compute(predictions=predictions, references=real)

loaded_wer = load('wer')

# predictions = ["this is the prediction yet, I dónt know", 'Pollo Loco']
# real = ["this is the prediction yet I dont know", 'pollo loco']

# get_wer(predictions, real, wer_model=loaded_wer, evaluate_accent=True, evaluate_punctuation=True)

#%% Run Benchmark
transcript = []
results_deepgram_nova = []
results_deepgram_enhanced = []
results_cartesia = []
results_elevenlabs = []

for i,tr in enumerate(transcripts):
    print(i,tr)
    transcript.append(tr)
    
    results_deepgram_nova.append(stt_deepgram(f'audios/mp3/AudioBenchmark_{i}.mp3', deepgram_client))
    results_deepgram_enhanced.append(stt_deepgram(f'audios/mp3/AudioBenchmark_{i}.mp3', deepgram_client, model_str='enhanced'))
    results_cartesia.append(stt_cartesia(f'audios/mp3/AudioBenchmark_{i}.mp3'))
    results_elevenlabs.append(stt_elevenlabs(f'audios/mp3/AudioBenchmark_{i}.mp3'))
    
results = pd.DataFrame({'Transcript':transcript, 
                        'results_Deepgram_Nova-3':results_deepgram_nova,
                        'Deepgram_Enhanced':results_deepgram_enhanced,
                        'Cartesia':results_cartesia,
                        'ElevenLabs':results_elevenlabs
                        })
results.to_csv('Results.csv')
results.to_clipboard()
    
    
#%%
    
nova = get_wer(results_deepgram_nova, transcript, evaluate_punctuation=False, remove=['¿¡'])
enhanced = get_wer(results_deepgram_enhanced, transcript, evaluate_punctuation=False, remove=['¿¡'])
cartesia = get_wer(results_cartesia, transcript, evaluate_punctuation=False, remove=['¿¡'])
elevenlabs = get_wer(results_elevenlabs, transcript, evaluate_punctuation=False, remove=['¿¡'])

nova, enhanced, cartesia, elevenlabs
#%%
    
nova = get_wer(results_deepgram_nova, transcript)
enhanced = get_wer(results_deepgram_enhanced, transcript)
cartesia = get_wer(results_cartesia, transcript)
elevenlabs = get_wer(results_elevenlabs, transcript)

nova, enhanced, cartesia, elevenlabs
#%%
    
nova = get_wer(results_deepgram_nova, transcript, evaluate_punctuation=False)
enhanced = get_wer(results_deepgram_enhanced, transcript, evaluate_punctuation=False)
cartesia = get_wer(results_cartesia, transcript, evaluate_punctuation=False)
elevenlabs = get_wer(results_elevenlabs, transcript, evaluate_punctuation=False)

nova, enhanced, cartesia, elevenlabs

#%%







