#!/usr/bin/env python
import argparse
import pathlib
import subprocess
import sys
from typing import Optional

import librosa
import numpy as np
import soundfile as sf
from music21 import converter


def preprocess_audio(input_path: pathlib.Path, target_sr: int = 22050) -> pathlib.Path:
    """Load audio, convert to mono, resample, normalize, and write to WAV.

    - ステレオ → モノラル
    - サンプリングレートを target_sr に統一
    - 振幅正規化（ピーク 0.95）
    """
    y, sr = librosa.load(str(input_path), sr=target_sr, mono=True)
    if y.size == 0:
        raise ValueError(f"Empty audio after loading: {input_path}")
    peak = np.max(np.abs(y))
    if peak > 0:
        y = 0.95 * y / peak
    out_path = input_path.with_suffix(".normalized.wav")
    sf.write(str(out_path), y, target_sr)
    return out_path


def run_basic_pitch(
    wav_path: pathlib.Path,
    output_dir: pathlib.Path,
) -> pathlib.Path:
    """Run Basic Pitch and return path to generated MIDI file.

    Basic Pitch は、入力音源と同じベース名にサフィックスを付けた .mid を生成する。
    例: build/foo.normalized.wav → build/foo.normalized_basic_pitch.mid
    """
    # ★ ここで初めて Basic Pitch を import（--score-only ではこの関数自体が呼ばれない）
    try:
        from basic_pitch.inference import predict_and_save
        from basic_pitch import ICASSP_2022_MODEL_PATH
    except Exception as exc:  # ImportError でも NameError でもまとめて捕捉
        raise RuntimeError(
            "Basic Pitch (basic_pitch) をインポートできなかったため、"
            "Audio→MIDI ステップを実行できません。\n"
            "・Audio→MIDI が不要な場合は --score-only を使ってください。\n"
            "・フルパイプラインを使う場合は basic_pitch のインストール／バージョンを確認してください。"
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)

    stem = wav_path.stem  # 例: "foo.normalized"

    # Basic Pitch が生成する既存ファイルがあると OSError を投げてスキップするので、
    # 実行前に掃除しておく。
    for ext in ("mid", "npz", "csv", "wav"):
        p = output_dir / f"{stem}_basic_pitch.{ext}"
        if p.exists():
            p.unlink()

    # Basic Pitch 実行
    predict_and_save(
        audio_path_list=[str(wav_path)],
        output_directory=str(output_dir),
        save_midi=True,
        sonify_midi=False,
        save_model_outputs=False,
        save_notes=False,  # ノート CSV は使わない
        model_or_model_path=ICASSP_2022_MODEL_PATH,
    )

    # 出力された .mid を探す（通常は <stem>_basic_pitch.mid）
    preferred = output_dir / f"{stem}_basic_pitch.mid"
    if preferred.exists():
        midi_path = preferred
    else:
        candidates = sorted(output_dir.glob(f"{stem}*.mid"))
        if not candidates:
            raise FileNotFoundError(
                f"Expected MIDI file for {wav_path} not found under {output_dir}. "
                f"(tried '{preferred.name}' and pattern '{stem}*.mid')"
            )
        midi_path = candidates[0]

    return midi_path


def run_musescore_convert(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    musescore_cmd: str,
) -> bool:
    """汎用 MuseScore CLI ラッパー: 1 ファイル変換を行う。

    - 成功: True を返す
    - 失敗: WARNING を表示して False を返す
    """
    cmd = [musescore_cmd, "-o", str(output_path), str(input_path)]
    print(f"[INFO] Running MuseScore CLI: {' '.join(cmd)}", file=sys.stderr)
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        print(
            f"[WARNING] MuseScore command '{musescore_cmd}' not found; "
            f"skip conversion {input_path} → {output_path}. ({exc})",
            file=sys.stderr,
        )
        return False
    except subprocess.CalledProcessError as exc:
        # SIGSEGV を含む MuseScore 側のクラッシュもここに入る
        print(
            f"[WARNING] MuseScore CLI failed (exit={exc.returncode}) "
            f"when converting {input_path} → {output_path}; skip this conversion.",
            file=sys.stderr,
        )
        return False
    return True


def midi_to_musicxml(
    midi_path: pathlib.Path,
    musicxml_path: pathlib.Path,
    musescore_cmd: str,
) -> pathlib.Path:
    """MIDI → MusicXML を生成する。

    優先:
      1. MuseScore CLI に MIDI を直接渡して MusicXML を生成
      2. 失敗した場合は music21 にフォールバックして MusicXML を生成
    """
    # 1. まず mscore に MIDI を渡す
    if run_musescore_convert(midi_path, musicxml_path, musescore_cmd):
        return musicxml_path

    # 2. ダメなら music21 でフォールバック
    print(
        "[INFO] Falling back to music21 for MIDI → MusicXML conversion.",
        file=sys.stderr,
    )
    score = converter.parse(str(midi_path))
    score.write("musicxml", fp=str(musicxml_path))
    return musicxml_path


def musicxml_to_pdf(
    musicxml_path: pathlib.Path,
    pdf_path: pathlib.Path,
    musescore_cmd: str,
) -> Optional[pathlib.Path]:
    """MusicXML → PDF を MuseScore CLI で行う。

    - 失敗してもパイプライン全体は止めず、None を返すだけ。
    """
    ok = run_musescore_convert(musicxml_path, pdf_path, musescore_cmd)
    if not ok:
        return None
    return pdf_path


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audio → MIDI → MusicXML → PDF pipeline using Basic Pitch, music21, and MuseScore.\n"
            "通常:\n"
            "  audio2score.py build/foo.wav --output-dir build\n"
            "既存の normalized MIDI (build/foo.normalized_basic_pitch.mid) から "
            "MusicXML / PDF だけ作る:\n"
            "  audio2score.py build/foo.wav --output-dir build --score-only\n"
        )
    )
    parser.add_argument("audio", help="Input WAV file (produced by ffmpeg etc.)")
    parser.add_argument(
        "--backend",
        default="basic-pitch",
        choices=["basic-pitch"],
        help="Transcription backend (currently only 'basic-pitch').",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="build",
        help="Directory where MIDI / MusicXML / PDF will be written.",
    )
    parser.add_argument(
        "--musescore-cmd",
        type=str,
        default="mscore",
        help="MuseScore CLI command (e.g., 'mscore', 'musescore4').",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Do not generate PDF (only MIDI and MusicXML).",
    )
    parser.add_argument(
        "--score-only",
        action="store_true",
        help=(
            "既に Basic Pitch によって生成された normalized MIDI "
            "（build/<name>.normalized_basic_pitch.mid）を再利用し、"
            "MusicXML / PDF だけを生成する。"
            "Audio → MIDI 推論は行わない。"
        ),
    )
    args = parser.parse_args(argv)

    input_path = pathlib.Path(args.audio).expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f"Input audio not found: {input_path}")

    output_dir = pathlib.Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    normalized: Optional[pathlib.Path] = None
    midi_path: pathlib.Path

    if args.score_only:
        # Audio → MIDI は既に完了している前提。
        # build/foo.wav → build/foo.normalized.wav → build/foo.normalized_basic_pitch.mid
        normalized = input_path.with_suffix(".normalized.wav")
        stem = normalized.stem  # 例: "foo.normalized"
        candidate = output_dir / f"{stem}_basic_pitch.mid"
        if not candidate.exists():
            raise SystemExit(
                f"Expected normalized MIDI not found: {candidate}\n"
                "先に full パイプラインで normalized MIDI を生成してください。\n"
                "例: audio2score.py build/foo.wav --output-dir build"
            )
        midi_path = candidate
    else:
        # 1. 前処理 (Audio WAV → normalized WAV)
        normalized = preprocess_audio(input_path)
        # 2. Audio → MIDI (Basic Pitch)
        midi_path = run_basic_pitch(normalized, output_dir)

    # 3. MIDI → MusicXML（優先: MuseScore / フォールバック: music21）
    musicxml_path = output_dir / f"{midi_path.stem}.musicxml"
    midi_to_musicxml(midi_path, musicxml_path, musescore_cmd=args.musescore_cmd)

    # 4. MusicXML → PDF（MuseScore、失敗したらスキップ）
    pdf_path: Optional[pathlib.Path] = None
    if not args.no_pdf:
        pdf_path = musicxml_to_pdf(
            musicxml_path,
            output_dir / f"{midi_path.stem}.pdf",
            musescore_cmd=args.musescore_cmd,
        )

    print("=== Audio to score conversion completed ===")
    print(f"Input audio : {input_path}")
    if normalized is not None:
        print(f"Normalized  : {normalized}")
    print(f"MIDI        : {midi_path}")
    print(f"MusicXML    : {musicxml_path}")
    if pdf_path is not None:
        print(f"PDF         : {pdf_path}")
    else:
        print("PDF         : (not generated)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
