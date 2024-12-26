import os
import queue
import io
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy import signal
from dotenv import load_dotenv

import torch
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline
)

import openai

# Pustaka TTS
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play

# Pustaka untuk waktu & tanggal
from datetime import datetime

# ------------------------------------------------------------------------------
# 1. Load API Key dari .env
# ------------------------------------------------------------------------------
load_dotenv()  # Memuat file .env
openai.api_key = os.getenv("OPENAI_API_KEY")  # Pastikan .env memuat OPENAI_API_KEY=sk-...

# ------------------------------------------------------------------------------
# 2. Inisialisasi Model Whisper (untuk transkripsi)
# ------------------------------------------------------------------------------
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
)

# Paksa bahasa Indonesia (agar transkripsi condong ke Bahasa Indonesia)
forced_decoder_ids = processor.get_decoder_prompt_ids(
    language="Indonesian", task="transcribe"
)

# ------------------------------------------------------------------------------
# 3. Parameter Audio & VAD
# ------------------------------------------------------------------------------
sample_rate = 16000
block_size = 16000   # 1 detik per blok
silence_threshold = 0.01       # Ambang RMS untuk dianggap diam
required_silence_blocks = 2    # Diam berapa blok berturut-turut => "selesai bicara"
silence_count = 0

# Buffer menampung blok audio selama user bicara
audio_buffer = []

# ------------------------------------------------------------------------------
# 4. Rolling window untuk menampilkan spectrogram (5 detik)
# ------------------------------------------------------------------------------
rolling_window = np.array([], dtype=np.float32)
rolling_duration = 5  # detik

# ------------------------------------------------------------------------------
# 5. Setup input stream + queue (sounddevice)
# ------------------------------------------------------------------------------
q = queue.Queue()

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    q.put(indata.copy())

# ------------------------------------------------------------------------------
# 6. Persiapan Matplotlib untuk spectrogram
# ------------------------------------------------------------------------------
plt.ion()  # mode interaktif
fig, ax = plt.subplots(figsize=(8, 4))

print("=== Mulai merekam (rolling spectrogram). Tekan Ctrl+C untuk berhenti. ===")
print("=== Dapat mengatakan 'jam berapa', 'tanggal berapa', 'halo asisten', 'menurut anda', dll. ===")

try:
    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        callback=audio_callback,
        blocksize=block_size
    ):
        while True:
            # Ambil 1 detik audio dari queue
            block_audio = q.get().flatten()

            # ------------------------------------------------------------------
            # 6a. Update rolling window (untuk spectrogram 5 detik terakhir)
            # ------------------------------------------------------------------
            rolling_window = np.concatenate((rolling_window, block_audio))
            max_len = int(rolling_duration * sample_rate)  # 5 detik => 80000 sample
            if len(rolling_window) > max_len:
                # pangkas bagian depan
                rolling_window = rolling_window[-max_len:]

            # ------------------------------------------------------------------
            # 6b. Plot rolling spectrogram
            # ------------------------------------------------------------------
            f, t, Sxx = signal.spectrogram(
                rolling_window,
                fs=sample_rate,
                nperseg=512,
                noverlap=256
            )
            ax.clear()
            extent = [0, rolling_duration, f[0], f[-1]]
            ax.imshow(
                10 * np.log10(Sxx),
                aspect='auto',
                origin='lower',
                cmap='magma',
                extent=extent
            )
            ax.set_ylabel("Frekuensi [Hz]")
            ax.set_xlabel("Waktu [detik] (window 5 detik)")
            ax.set_title("Spectrogram Bergeser Real-time (5 detik)")
            plt.draw()
            plt.pause(0.001)

            # ------------------------------------------------------------------
            # 7. VAD Sederhana (RMS threshold)
            # ------------------------------------------------------------------
            rms = np.sqrt(np.mean(block_audio ** 2))
            if rms < silence_threshold:
                silence_count += 1
            else:
                silence_count = 0

            # Simpan blok audio di buffer
            audio_buffer.append(block_audio)

            # Jika sudah diam beberapa blok => transkripsi
            if silence_count >= required_silence_blocks and len(audio_buffer) > 0:
                # Gabungkan seluruh blok di buffer
                full_audio = np.concatenate(audio_buffer)

                # Jalankan Whisper
                audio_input = {"array": full_audio, "sampling_rate": sample_rate}
                result = asr_pipeline(
                    audio_input,
                    generate_kwargs={"forced_decoder_ids": forced_decoder_ids}
                )

                # Hasil transkripsi
                transcribed_text = result["text"]
                # Opsional: ganti "Terima kasih." => "Silakan berbicara."
                transcribed_text = transcribed_text.replace(
                    "Terima kasih.", "Silakan berbicara."
                )
                print(">>> (Hasil Transkripsi) :", transcribed_text)

                # Lowercase untuk memudahkan pencarian kata kunci
                lower_text = transcribed_text.lower()

                # ------------------------------------------------------------------
                # 8a. Jika menanyakan waktu/tanggal
                # ------------------------------------------------------------------
                if "jam berapa sekarang" in lower_text:
                    # Dapatkan waktu sekarang
                    now = datetime.now()
                    current_time = now.strftime("%H:%M:%S")  # 24-jam format
                    answer = f"Sekarang jam {current_time}"
                    print(">>> (Jawaban Waktu) :", answer)

                    # TTS
                    try:
                        tts = gTTS(text=answer, lang='id')
                        buf = io.BytesIO()
                        tts.write_to_fp(buf)
                        buf.seek(0)
                        tts_audio = AudioSegment.from_file(buf, format="mp3")
                        play(tts_audio)
                    except Exception as e:
                        print("Gagal memutar suara TTS:", str(e))

                elif "tanggal berapa" in lower_text:
                    # Dapatkan tanggal sekarang
                    now = datetime.now()
                    current_date = now.strftime("%Y-%m-%d")
                    answer = f"Hari ini tanggal {current_date}"
                    print(">>> (Jawaban Tanggal) :", answer)

                    # TTS
                    try:
                        tts = gTTS(text=answer, lang='id')
                        buf = io.BytesIO()
                        tts.write_to_fp(buf)
                        buf.seek(0)
                        tts_audio = AudioSegment.from_file(buf, format="mp3")
                        play(tts_audio)
                    except Exception as e:
                        print("Gagal memutar suara TTS:", str(e))

                # ------------------------------------------------------------------
                # 8b. Jika kata kunci lain (untuk ChatGPT)
                # ------------------------------------------------------------------
                else:
                    # Misal kata kunci untuk ChatGPT
                    keywords = ["halo asisten", "menurut anda", "jelaskan", "bagaimana", "apakah", "tulislah", "buatlah"]
                    if any(kw in lower_text for kw in keywords):
                        try:
                            response = openai.ChatCompletion.create(
                                model="gpt-4o-mini",  # Sesuaikan dengan model Anda
                                messages=[
                                    {
                                        "role": "system",
                                        "content": (
                                            "Kamu adalah asisten AI yang sangat membantu "
                                            "dan akan menjawab dalam bahasa Indonesia."
                                        )
                                    },
                                    {
                                        "role": "user",
                                        "content": transcribed_text
                                    }
                                ]
                            )
                            chatgpt_answer = response["choices"][0]["message"]["content"]
                            print(">>> (Jawaban ChatGPT) :", chatgpt_answer)

                            # TTS
                            try:
                                tts = gTTS(text=chatgpt_answer, lang='id')
                                buf = io.BytesIO()
                                tts.write_to_fp(buf)
                                buf.seek(0)
                                tts_audio = AudioSegment.from_file(buf, format="mp3")
                                play(tts_audio)
                            except Exception as e:
                                print("Gagal memutar suara TTS:", str(e))

                        except Exception as e:
                            print("Gagal memanggil ChatGPT:", str(e))
                    # else:
                    #   Tidak ada kata kunci ChatGPT, kita diam saja

                # Reset buffer & counter
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
