"""
About the Gulf Arabic Conversational Telephone Speech Corpus

  The Gulf Arabic Conversational corpus consists of contains roughly 
  2,800 min of spontaneous telephone conversations in Colloquial 
  Gulf Arabic. The datasets are:
  Speech : LDC2006S43
  Transcripts : LDC2006T15
"""

from decimal import Decimal
from pathlib import Path
from logging import info
from re import sub
from string import punctuation
from typing import Dict, Optional, Union
import pdb
from tqdm.auto import tqdm

from lhotse import Recording, RecordingSet, SupervisionSegment, SupervisionSet
from lhotse.qa import fix_manifests, validate_recordings_and_supervisions
from lhotse.utils import Pathlike, check_and_rglob


def prepare_callhome_gulf(
    audio_dir: Pathlike,
    transcript_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    text_cleaning: bool = True,
    num_jobs: int = 10,
) -> Dict[str, Union[RecordingSet, SupervisionSet]]:
    """
    Prepare manifests for the Callhome Gulf Arabic Corpus
    We create two manifests: one with recordings, and the other one with text
    supervisions.

    :param audio_dir: Path to ``LDC2006S43`` package.
    :param transcript_dir: Path to the ``LDC2006T15`` content
    :param output_dir: Directory where the manifests should be written. Can be omitted
        to avoid writing.
    :param text_cleaning: Bool, if True, basic text cleaning is performed.
    :param num_jobs: int, the number of jobs to use for parallel processing.
    :return: A dict with manifests. The keys are: ``{'recordings', 'supervisions'}``.
    """
    audio_dir = Path(audio_dir)
    transcript_dir = Path(transcript_dir)

    manifests = {}

    for split in ["devtest", "train2c", "train1c"]:
        info(f"Processing Callhome Gulf subset: {split}")
        recordings = RecordingSet.from_dir(
            (audio_dir / "audio" / split), pattern="*.sph", num_jobs=num_jobs
        )
        transcript_paths = check_and_rglob(
            transcript_dir /
            f"transc/{split}",
            "*.txt",
        )

        # TODO: Add text normalization.
        supervisions = []
        for p in transcript_paths:
            idx = 0
            for line in p.read_text().splitlines():
                line = line.strip()
                if not line or len(line.split('\t')) < 4:
                    continue
                recording_id = p.stem
                # example line:
                # [3.2701] [5.4100]	B:	ألو (ضجة)	>alaw (Djp)
                time, spk, text, text_bw = line.split('\t')
                time = sub(r"[\[\]]", "", time)
                start, end = time.split()
                spk = spk.replace(":", "")
                duration = float(Decimal(end) - Decimal(start))
                if text_cleaning is True:
                    text = cleaning(text)

                if duration <= 0 or text == '':
                    continue

                start = float(start)
                supervisions.append(
                    SupervisionSegment(
                        id=f"{recording_id}_{idx}",
                        recording_id=recording_id,
                        start=start,
                        duration=duration,
                        speaker=f"{recording_id}_{spk}",
                        text=text,
                    )
                )
                idx += 1

        supervisions = SupervisionSet.from_segments(supervisions)

        recordings, supervisions = fix_manifests(recordings, supervisions)
        validate_recordings_and_supervisions(recordings, supervisions)

        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            recordings.to_file(
                output_dir / f"gulf_recordings_{split}.jsonl.gz"
            )
            supervisions.to_file(
                output_dir / f"gulf_supervisions_{split}.jsonl.gz"
            )

        manifests[split] = {"recordings": recordings,
                            "supervisions": supervisions}

    return manifests


def remove_diacritics(text: str) -> str:
    # https://unicode-table.com/en/blocks/arabic/
    return sub(r"[\u064B-\u0652\u06D4\u0670\u0674\u06D5-\u06ED]+", "", text)


def remove_punctuations(text: str) -> str:
    """This function  removes all punctuations except the verbatim"""

    arabic_punctuations = """﴿﴾`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ"""
    english_punctuations = punctuation
    # remove all non verbatim punctuations
    all_punctuations = set(arabic_punctuations + english_punctuations)

    for p in all_punctuations:
        if p in text:
            text = text.replace(p, " ")
    return text


def remove_non_alphanumeric(text: str) -> str:
    text = text.lower()
    return sub(r"[^\u0600-\u06FF\s\da-z]+", "", text)


def remove_extra_space(text: str) -> str:
    text = sub("\s+", " ", text)
    text = sub("\s+\.\s+", ".", text)
    return text


def remove_special(text: str) -> str:
    """
    This functions removes special tokens:
        (): tokens enclosed in single parentheses represent nonspeech
            events laughter, noise, etc 
        (()): tokens enclosed in double parentheses represent regions
            where the transcriber was unable to tell for sure what was said.
    """
    text = sub(r"\([\\\u0600-\u06FF]+\)", "", text)
    text = sub(r"\(\([\u0600-\u06FF\s]+\)\)", "", text)
    return text


def cleaning(text: str) -> str:
    text = remove_diacritics(text)
    text = remove_special(text)
    text = remove_punctuations(text)
    text = remove_non_alphanumeric(text)
    text = remove_extra_space(text)
    return text


if __name__ == "__main__":
    manifests = prepare_callhome_gulf('/alt-data/speech/asr_data/LDC/LDC2006S43',
                                      '/alt-data/speech/asr_data/LDC/LDC2006T15', '/home/local/QCRI/ahussein/data')
