import torch
import sounddevice as sd
import queue
import numpy as np
from faster_whisper import WhisperModel

# --------------------------------------------------------------------------------
# 1. Inisialisasi model
# --------------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if torch.cuda.is_available() else "int8"

model_size = "large-v3"
print(f"Device set to use {device}")

# Inisialisasi WhisperModel
model = WhisperModel(model_size, device=device, compute_type=compute_type)

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
                full_audio = np.concatenate(audio_buffer).astype(np.float32)

                # Buat input file sementara untuk transkripsi
                # print("=== Mendeteksi akhir percakapan, mulai transkripsi... ===")
                segments, info = model.transcribe(
                    full_audio, beam_size=5, language="id"
                )

                # Cetak hasil
                # print(f"Detected language: {info.language} (Probability: {info.language_probability})")
                for segment in segments:
                    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s]: {segment.text}")

                # Kosongkan buffer setelah transkripsi
                audio_buffer = []
                silence_count = 0

except KeyboardInterrupt:
    print("\n=== Perekaman dihentikan oleh user. ===")
except Exception as e:
    print("Terjadi error:", str(e))
