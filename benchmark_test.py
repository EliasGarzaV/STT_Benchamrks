#%%
# from evaluate import load
from pydub import AudioSegment
import os
import json
import requests
import pandas as pd
from dotenv import load_dotenv
import json
from deepgram import DeepgramClient
from elevenlabs.client import ElevenLabs
from openai import OpenAI
import time
load_dotenv()

DEEPGRAMG_API_KEY = os.getenv("DEEPGRAM_API_KEY")
CARTESIA_API_KEY = os.getenv("CARTESIA_KEY")
ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY')
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


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

# for i in range(len(transcripts)):
#     convert_opus_to_mp3(f'audios\opus\OPUS_{i}.opus', f'audios\mp3\AudioBenchmark_{i}.mp3')


#%%Deepgram API wrapper

def stt_deepgram(audio_path: str, client: DeepgramClient, model_str: str = "nova-3"):
    try:
        start_time = time.perf_counter()
        with open(audio_path, "rb") as audio_file:
            response = client.listen.v1.media.transcribe_file(
                request=audio_file.read(),
                model=model_str,
                language="es",
                smart_format=True,
            )
        end_time = time.perf_counter()

        total_latency = end_time - start_time
        response_json = json.loads(response.json())
        transcript = response_json["results"]["channels"][0]["alternatives"][0]["transcript"]

        processing_time = response_json.get("metadata", {}).get("processing_time")
        network_latency = total_latency - processing_time if processing_time else None

        latency_info = {
            "total_latency": total_latency,
            "processing_time": processing_time,
            "network_latency": network_latency,
        }

        return transcript, latency_info
    except Exception as e:
        raise e

deepgram_client = DeepgramClient(api_key=DEEPGRAMG_API_KEY)


# stt_deepgram('audios/mp3/AudioBenchmark_1.mp3', deepgram_client)

#%% Cartesia API wrapper

def stt_cartesia(audio_path: str):
    try:
        url = "https://api.cartesia.ai/stt"
        headers = {
            "Authorization": f"Bearer {CARTESIA_API_KEY}",
            "Cartesia-Version": "2025-04-16",
        }
        files = {"file": open(audio_path, "rb")}
        data = {
            "model": "ink-whisper",
            "language": "es",
            "timestamp_granularities[]": "word",
        }

        start_time = time.perf_counter()
        response = requests.post(url, headers=headers, files=files, data=data)
        end_time = time.perf_counter()

        response.raise_for_status()
        result = response.json()

        total_latency = end_time - start_time

        transcript = result.get("text") or result.get("transcript", "")
        latency_info = {
            "total_latency": total_latency,
            "processing_time": None,
            "network_latency": None,
        }

        return transcript, latency_info
    except Exception as e:
        raise e

# stt_cartesia('audios/mp3/AudioBenchmark_1.mp3')

#%%Elevenlabs API Wrapper

def stt_elevenlabs(audio_path: str, client: ElevenLabs):
    try:
        start_time = time.perf_counter()
        with open(audio_path, "rb") as audio_file:
            transcription = client.speech_to_text.convert(
                file=audio_file,
                model_id="scribe_v1",
                tag_audio_events=True,
                language_code="spa",
                diarize=False,
            )
        end_time = time.perf_counter()

        total_latency = end_time - start_time
        transcript = transcription.dict().get("text", "")

        latency_info = {
            "total_latency": total_latency,
            "processing_time": None,
            "network_latency": None,
        }

        return transcript, latency_info
    except Exception as e:
        raise e

elevenlabs = ElevenLabs(api_key=ELEVENLABS_API_KEY)

# stt_elevenlabs('audios/mp3/AudioBenchmark_1.mp3', elevenlabs)

#%% OPENAI Transcribe wrapper

def stt_openai(audio_path: str, client: OpenAI):
    try:
        start_time = time.perf_counter()
        with open(audio_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="gpt-4o-transcribe",
                file=audio_file,
            )
        end_time = time.perf_counter()

        total_latency = end_time - start_time
        transcript = transcription.text

        latency_info = {
            "total_latency": total_latency,
            "processing_time": None,  # OpenAI doesn't return server-side timing
            "network_latency": None,
        }

        return transcript, latency_info
    except Exception as e:
        raise e

openai_client = OpenAI(api_key=OPENAI_API_KEY)

# stt_openai("audios\mp3\AudioBenchmark_0.mp3", openai_client)

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
results_openai = []
results_deepgram_nova_latency = []
results_deepgram_enhanced_latency = []
results_cartesia_latency = []
results_elevenlabs_latency = []
results_openai_latency = []

for i,tr in enumerate(transcripts):
    try:
        print(i,tr)
        transcript.append(tr)
        
        deepgram_nova = stt_deepgram(f'audios/mp3/AudioBenchmark_{i}.mp3', deepgram_client)
        results_deepgram_nova.append(deepgram_nova[0])
        results_deepgram_nova_latency.append(deepgram_nova[1]['total_latency'])
        
        dg_enh = stt_deepgram(f'audios/mp3/AudioBenchmark_{i}.mp3', deepgram_client, model_str='enhanced')
        results_deepgram_enhanced.append(dg_enh[0])
        results_deepgram_enhanced_latency.append(dg_enh[1]['total_latency'])
        
        car = stt_cartesia(f'audios/mp3/AudioBenchmark_{i}.mp3')
        results_cartesia.append(car[0])
        results_cartesia_latency.append(car[1]['total_latency'])
        
        ellabs = stt_elevenlabs(f'audios/mp3/AudioBenchmark_{i}.mp3', elevenlabs)
        results_elevenlabs.append(ellabs[0])
        results_elevenlabs_latency.append(ellabs[1]['total_latency'])
        
        op_ai = stt_openai(f'audios/mp3/AudioBenchmark_{i}.mp3', openai_client)
        results_openai.append(op_ai[0])
        results_openai_latency.append(op_ai[1]['total_latency'])
        
    except Exception as e:
        print(e)

#%%
    
    
results = pd.DataFrame({'Transcript':transcript, 
                        'results_Deepgram_Nova-3':results_deepgram_nova,
                        'Deepgram_Enhanced':results_deepgram_enhanced,
                        'Cartesia':results_cartesia,
                        'ElevenLabs':results_elevenlabs,
                        'OpenAI_Transcribe':results_openai
                        })

latency = pd.DataFrame({'Transcript':transcript, 
                        'results_Deepgram_Nova-3':results_deepgram_nova_latency,
                        'Deepgram_Enhanced':results_deepgram_enhanced_latency,
                        'Cartesia':results_cartesia_latency,
                        'ElevenLabs':results_elevenlabs_latency,
                        'OpenAI_Transcribe':results_openai_latency
                        })
results.to_csv('Results.csv')
latency.to_csv('Latency.csv')
results.to_clipboard()
    
    
#%%
    
nova = get_wer(results_deepgram_nova, transcript, evaluate_punctuation=False, remove=['¿¡'])
enhanced = get_wer(results_deepgram_enhanced, transcript, evaluate_punctuation=False, remove=['¿¡'])
cartesia = get_wer(results_cartesia, transcript, evaluate_punctuation=False, remove=['¿¡'])
elevenlabs = get_wer(results_elevenlabs, transcript, evaluate_punctuation=False, remove=['¿¡'])
openai = get_wer(results_openai, transcript, evaluate_punctuation=False, remove=['¿¡'])

nova, enhanced, cartesia, elevenlabs
#%%
    
nova = get_wer(results_deepgram_nova, transcript)
enhanced = get_wer(results_deepgram_enhanced, transcript)
cartesia = get_wer(results_cartesia, transcript)
elevenlabs = get_wer(results_elevenlabs, transcript)
openai = get_wer(results_openai, transcript)


nova, enhanced, cartesia, elevenlabs
#%%
    
nova = get_wer(results_deepgram_nova, transcript, evaluate_punctuation=False)
enhanced = get_wer(results_deepgram_enhanced, transcript, evaluate_punctuation=False)
cartesia = get_wer(results_cartesia, transcript, evaluate_punctuation=False)
elevenlabs = get_wer(results_elevenlabs, transcript, evaluate_punctuation=False)
openai = get_wer(results_openai, transcript, evaluate_punctuation=False)

nova, enhanced, cartesia, elevenlabs

#%%







