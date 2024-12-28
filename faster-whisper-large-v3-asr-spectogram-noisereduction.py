import queue
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from faster_whisper import WhisperModel
import sounddevice as sd
import noisereduce as nr
import torch

# -------------------------------------------------------------------
# 1. Inisialisasi Model Faster Whisper
# -------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if torch.cuda.is_available() else "int8"

model_size = "large-v3"
print(f"Device set to use {device}")

model = WhisperModel(model_size, device=device, compute_type=compute_type)

# -------------------------------------------------------------------
# 2. Parameter Audio & VAD
# -------------------------------------------------------------------
sample_rate = 16000
block_size = 16000  # 1 detik per blok
silence_threshold = 0.01      # Ambang RMS untuk dianggap diam
required_silence_blocks = 2   # Berapa blok diam berurutan hingga dianggap selesai bicara
silence_count = 0

# Buffer untuk menampung audio selama user bicara
audio_buffer = []

# Rolling window untuk menampilkan spectrogram (5 detik)
rolling_window = np.array([], dtype=np.float32)
rolling_duration = 5  # detik

# -------------------------------------------------------------------
# 3. Setup input stream + queue
# -------------------------------------------------------------------
q = queue.Queue()

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    q.put(indata.copy())

# -------------------------------------------------------------------
# 4. Persiapan Matplotlib untuk spectrogram
# -------------------------------------------------------------------
plt.ion()  # mode interaktif, supaya plot bisa di-update
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

print("=== Mulai merekam (scrolling spectrogram). Tekan Ctrl+C untuk berhenti. ===")

try:
    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        callback=audio_callback,
        blocksize=block_size
    ):
        while True:
            # Ambil 1 blok audio dari queue (1 detik)
            block_audio = q.get().flatten()

            # ---------------------------------------------------------------
            # 4a. Update rolling window (untuk menampilkan 5 detik terakhir)
            # ---------------------------------------------------------------
            rolling_window = np.concatenate((rolling_window, block_audio))
            max_len = int(rolling_duration * sample_rate)  # 5 * 16000 = 80000
            if len(rolling_window) > max_len:
                rolling_window = rolling_window[-max_len:]  # potong depan

            # ---------------------------------------------------------------
            # 4b. Plot spectrogram sebelum noise reduction
            # ---------------------------------------------------------------
            f, t, Sxx = signal.spectrogram(
                rolling_window,
                fs=sample_rate,
                nperseg=512,
                noverlap=256
            )
            ax1.clear()
            ax1.imshow(
                10 * np.log10(Sxx),
                aspect='auto',
                origin='lower',
                cmap='magma',
                extent=[0, rolling_duration, f[0], f[-1]]
            )
            ax1.set_ylabel("Frekuensi [Hz]")
            ax1.set_title("Spectrogram Sebelum Noise Reduction")

            # ---------------------------------------------------------------
            # 4c. Noise reduction pada blok audio
            # ---------------------------------------------------------------
            reduced_audio = nr.reduce_noise(
                y=block_audio, sr=sample_rate, prop_decrease=0.8
            )

            # ---------------------------------------------------------------
            # 4d. Plot spectrogram setelah noise reduction
            # ---------------------------------------------------------------
            f_reduced, t_reduced, Sxx_reduced = signal.spectrogram(
                reduced_audio,
                fs=sample_rate,
                nperseg=512,
                noverlap=256
            )
            ax2.clear()
            ax2.imshow(
                10 * np.log10(Sxx_reduced),
                aspect='auto',
                origin='lower',
                cmap='viridis',
                extent=[0, rolling_duration, f[0], f[-1]]
            )
            ax2.set_ylabel("Frekuensi [Hz]")
            ax2.set_xlabel("Waktu [detik] (window 5 detik)")
            ax2.set_title("Spectrogram Setelah Noise Reduction")

            plt.draw()
            plt.pause(0.001)

            # ---------------------------------------------------------------
            # 5. VAD Sederhana (berbasis threshold RMS)
            # ---------------------------------------------------------------
            rms = np.sqrt(np.mean(block_audio ** 2))
            if rms < silence_threshold:
                silence_count += 1
            else:
                silence_count = 0

            # Selama masih ada suara, blok audio dikumpulkan ke audio_buffer
            audio_buffer.append(reduced_audio)

            # ---------------------------------------------------------------
            # 6. Jika diam berurutan >= required_silence_blocks, transkripsi
            # ---------------------------------------------------------------
            if silence_count >= required_silence_blocks and len(audio_buffer) > 0:
                # print("=== Terdeteksi akhir bicara, mulai transkripsi... ===")

                # Gabungkan semua blok jadi satu array
                full_audio = np.concatenate(audio_buffer)

                # Transkripsi menggunakan Faster Whisper
                segments, info = model.transcribe(full_audio, beam_size=5, language="id")

                # Cetak hasil transkripsi
                # print(f"Detected language: {info.language} (Probability: {info.language_probability})")
                for segment in segments:
                    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s]: {segment.text}")

                # Reset buffer bicara & counter
                audio_buffer = []
                silence_count = 0

except KeyboardInterrupt:
    print("\n=== Perekaman dihentikan oleh user. ===")
    plt.ioff()
    plt.show()
except Exception as e:
    print("Terjadi error:", str(e))
    plt.ioff()
    plt.show()
