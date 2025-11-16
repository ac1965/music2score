VENV := .venv

# pyenv 3.10 系を明示的に指定（環境に合わせて書き換え）
PYTHON_BASE ?= ~/.pyenv/versions/3.10.14/bin/python

PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

# INPUT:
# - 任意のオーディオファイル（aifc/wav/mp3/flac/m4a 等）
# - 例: ../env/input/どこにいるのだろう.aifc
INPUT ?= input.wav

OUTDIR ?= build

# ファイル名部分だけ抜き出して WAV を生成
BASENAME := $(basename $(notdir $(INPUT)))
WAV_INPUT := $(OUTDIR)/$(BASENAME).wav

# macOS / MuseScore 用（環境に応じて上書き可）
MUSESCORE_CMD ?= mscore

RUN_ARGS ?=

.PHONY: all setup score score-full score-existing score-no-pdf run clean

all: setup

$(VENV)/bin/activate: requirements.txt
	$(PYTHON_BASE) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	touch $(VENV)/bin/activate

setup: $(VENV)/bin/activate

$(OUTDIR):
	mkdir -p $(OUTDIR)

# 任意形式 → WAV への変換ステップ（ffmpeg 必須）
# 例:
#   INPUT = ../env/input/どこにいるのだろう.aifc
#   → WAV_INPUT = build/どこにいるのだろう.wav
$(WAV_INPUT): $(OUTDIR) $(INPUT)
	ffmpeg -y -i "$(INPUT)" -ac 1 -ar 44100 "$(WAV_INPUT)"

# ★ score = フルパイプライン（任意形式 → WAV → MIDI → MusicXML → PDF）
# 例: make score INPUT=../env/input/どこにいるのだろう.aifc
score: score-full

# 実体: フルパイプライン
score-full: setup $(WAV_INPUT)
	$(PYTHON) audio2score.py $(WAV_INPUT) \
		--output-dir $(OUTDIR) \
		--musescore-cmd $(MUSESCORE_CMD)

# 既存の normalized MIDI を再利用し、MusicXML / PDF のみ生成
# 例: make score-existing INPUT=../env/input/どこにいるのだろう.aifc
score-existing: setup $(WAV_INPUT)
	$(PYTHON) audio2score.py $(WAV_INPUT) \
		--output-dir $(OUTDIR) \
		--musescore-cmd $(MUSESCORE_CMD) \
		--score-only

# PDF は作らず、MIDI / MusicXML まで
# 例: make score-no-pdf INPUT=../env/input/どこにいるのだろう.aifc
score-no-pdf: setup $(WAV_INPUT)
	$(PYTHON) audio2score.py $(WAV_INPUT) \
		--output-dir $(OUTDIR) \
		--musescore-cmd $(MUSESCORE_CMD) \
		--no-pdf

# 任意の追加引数を渡したいときに使う汎用ターゲット
# 例: make run INPUT=../env/input/foo.mp3 RUN_ARGS="--no-pdf"
run: setup $(WAV_INPUT)
	$(PYTHON) audio2score.py $(WAV_INPUT) \
		--output-dir $(OUTDIR) \
		--musescore-cmd $(MUSESCORE_CMD) \
		$(RUN_ARGS)

clean:
	rm -rf $(VENV) $(OUTDIR) *.normalized.wav
