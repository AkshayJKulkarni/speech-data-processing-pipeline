# Speech Data Processing & Annotation Pipeline

An end-to-end pipeline for ingesting audio/video, processing speech, and generating
structured annotations — built to production engineering standards.

---

## What This Pipeline Does

Takes raw audio or video as input and produces structured annotation files (JSON/CSV)
containing transcriptions, speaker segments, and emotion labels — the kind of labeled
datasets used to train production speech models.

```
Raw Audio/Video
      │
      ▼
  Acquisition        ← Download from URL / YouTube / local file
      │
      ▼
  Preprocessing      ← Resample, denoise, normalize, VAD
      │
      ▼
  Transcription      ← Whisper speech-to-text
      │
      ▼
  Diarization        ← Who spoke when (pyannote)
      │
      ▼
  Emotion            ← Emotion per speaker segment
      │
      ▼
  Annotation Export  ← Structured JSON / CSV output
```

---

## Project Structure

```
├── data/
│   ├── raw/             # Original input files (gitignored)
│   ├── processed/       # Cleaned audio (gitignored)
│   └── output/          # Final annotations (gitignored)
├── src/
│   ├── acquisition/     # Audio/video ingestion
│   ├── preprocessing/   # Audio cleaning & normalization
│   ├── transcription/   # Speech-to-text
│   ├── diarization/     # Speaker diarization
│   ├── emotion/         # Emotion classification
│   ├── annotation/      # Output generation
│   ├── logger.py        # Centralized logging
│   └── config_loader.py # YAML config reader
├── configs/
│   ├── config.yaml      # Pipeline settings
│   └── models.yaml      # Model settings
├── tests/               # Unit tests
├── scripts/             # Utility scripts
├── logs/                # Runtime logs (gitignored)
├── main.py              # Pipeline entry point
├── requirements.txt
└── .gitignore
```

---

## Setup

```bash
# 1. Clone the repo
git clone <repo-url>
cd Speech-Data-Processing-Pipeline

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the pipeline
python main.py
```

> Note: Some models (pyannote, speechbrain) require a Hugging Face token.
> Set it as an environment variable: `HF_TOKEN=your_token_here`

---

## Configuration

All pipeline behavior is controlled via YAML files in `configs/`:

- `config.yaml` — paths, audio settings, log level
- `models.yaml` — model selection and inference device (cpu/cuda)

---

## Tech Stack

| Stage | Library |
|---|---|
| Audio I/O | librosa, soundfile, pydub |
| Transcription | OpenAI Whisper |
| Diarization | pyannote.audio |
| Emotion | SpeechBrain |
| Deep Learning | PyTorch, Transformers |
| Media Acquisition | yt-dlp |

---

## License

MIT
