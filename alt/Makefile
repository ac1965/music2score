VENV := .venv
PYTHON_BASE ?= python3
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

INPUT ?= input.wav
OUTDIR ?= build

# macOS / MuseScore 用（環境に応じて上書き可）
MUSESCORE_CMD ?= mscore

RUN_ARGS ?=

.PHONY: all setup score score-full score-no-pdf run clean

all: setup

$(VENV)/bin/activate: requirements.txt
	$(PYTHON_BASE) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	touch $(VENV)/bin/activate

setup: $(VENV)/bin/activate

# 既存の normalized MIDI を再利用し、MusicXML / PDF のみ生成
score: setup
	$(PYTHON) audio2score.py $(INPUT) \
		--output-dir $(OUTDIR) \
		--musescore-cmd $(MUSESCORE_CMD) \
		--score-only

# フルパイプライン（Audio → MIDI → MusicXML → PDF）
score-full: setup
	$(PYTHON) audio2score.py $(INPUT) \
		--output-dir $(OUTDIR) \
		--musescore-cmd $(MUSESCORE_CMD)

# PDF は作らず、MIDI / MusicXML まで
score-no-pdf: setup
	$(PYTHON) audio2score.py $(INPUT) \
		--output-dir $(OUTDIR) \
		--musescore-cmd $(MUSESCORE_CMD) \
		--no-pdf

# 任意の追加引数を渡したいときに使う汎用ターゲット
run: setup
	$(PYTHON) audio2score.py $(INPUT) \
		--output-dir $(OUTDIR) \
		--musescore-cmd $(MUSESCORE_CMD) \
		$(RUN_ARGS)

clean:
	rm -rf $(VENV) $(OUTDIR) *.normalized.wav
