import torch
import sounddevice as sd
import queue
import numpy as np
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# --------------------------------------------------------------------------------
# 1. Inisialisasi model
# --------------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"
print(f"Device set to use {device}")

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True
).to(device)

processor = AutoProcessor.from_pretrained(model_id)

asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    device=device,
    torch_dtype=torch_dtype,
    return_timestamps=True,
)

# Misal kita pakai bahasa Indonesia
forced_decoder_ids = processor.get_decoder_prompt_ids(
    language="Indonesian", task="transcribe"
)

# --------------------------------------------------------------------------------
# 2. Setup untuk input mikrofon
# --------------------------------------------------------------------------------
q = queue.Queue()
sample_rate = 16000

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    q.put(indata.copy())

# --------------------------------------------------------------------------------
# 3. VAD sederhana & transkripsi hanya setelah selesai bicara
# --------------------------------------------------------------------------------

# Pengaturan untuk 'deteksi diam'
silence_threshold = 0.01      # Ambang RMS (semakin besar => lebih sensitif)
required_silence_blocks = 2   # Jumlah blok berturut-turut yang dianggap diam
block_size = 16000            # 1 detik per blok (16000 sampel, 16 kHz)

# Buffer audio menampung semua blok saat kita berbicara
audio_buffer = []
silence_count = 0  # Counter berapa kali blok dikategorikan 'diam' berturut-turut

print("=== Mulai merekam (mode tunggu diam). Tekan Ctrl+C untuk berhenti. ===")

try:
    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        callback=audio_callback,
        blocksize=block_size
    ):
        while True:
            # Ambil 1 blok audio
            audio_data = q.get().flatten()

            # Hitung energi/rata-rata RMS (root mean square)
            rms = np.sqrt(np.mean(audio_data**2))

            # Apakah di bawah threshold => dianggap diam
            if rms < silence_threshold:
                silence_count += 1
            else:
                silence_count = 0

            # Tambahkan blok ke audio_buffer
            audio_buffer.append(audio_data)

            # Jika sudah diam beberapa blok berturut-turut -> transkripsi
            if silence_count >= required_silence_blocks and len(audio_buffer) > 0:
                # Gabungkan semua blok jadi satu array
                full_audio = np.concatenate(audio_buffer)
                
                # Buat input dictionary
                audio_input = {
                    "array": full_audio,
                    "sampling_rate": sample_rate
                }

                # Lakukan transkripsi
                print("=== Mendeteksi akhir percakapan, mulai transkripsi... ===")
                result = asr_pipeline(
                    audio_input,
                    generate_kwargs={"forced_decoder_ids": forced_decoder_ids}
                )

                # Cetak hasil
                print(">>>", result["text"])

                # Kosongkan buffer setelah transkripsi
                audio_buffer = []
                silence_count = 0

except KeyboardInterrupt:
    print("\n=== Perekaman dihentikan oleh user. ===")
except Exception as e:
    print("Terjadi error:", str(e))
