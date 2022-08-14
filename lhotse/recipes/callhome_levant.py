"""
About the Leventine Arabic Conversational Telephone Speech Corpus

  The Levantine Arabic QT Training Data Set 5 corpus contains 1,660 
  calls totalling approximately 250 hours. These calls were collected 
  between 2003 and 2005. 
  This corpus is the combination of previous training data sets, 
  including Arabic CTS Levantine Fisher Training Data Set 3, Speech 
  (LDC2005S07), Arabic CTS Levantine Fisher Training Data Set 3, 
  Transcripts (LDC2005T03), and Levantine Arabic QT Training Data 
  Set 4, Speech and Transcripts (LDC2005S14). More than half of the 
  speakers are Lebanese, the others are Jordanian, Palestinian, 
  and Syrian. 
The datasets are:
Speech : LDC2006S29
Transcripts : LDC2006T07

"""

from decimal import Decimal
from glob import glob
from pathlib import Path
from logging import info
from re import sub
import os
from string import punctuation
from typing import Dict, Optional, Union
import pdb
from tqdm.auto import tqdm

from lhotse import Recording, RecordingSet, SupervisionSegment, SupervisionSet
from lhotse.qa import fix_manifests, validate_recordings_and_supervisions
from lhotse.utils import Pathlike, check_and_rglob


def prepare_callhome_levant(
    audio_dir: Pathlike,
    transcript_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    text_cleaning: bool = True,
    num_jobs: int = 2,
) -> Dict[str, Union[RecordingSet, SupervisionSet]]:
    """
    Prepare manifests for the Callhome Gulf Arabic Corpus
    We create two manifests: one with recordings, and the other one with text
    supervisions.

    :param audio_dir: Path to ``LDC2006S45`` package.
    :param transcript_dir: Path to the ``LDC2006T16`` content
    :param output_dir: Directory where the manifests should be written. Can be omitted
        to avoid writing.
    :param text_cleaning: Bool, if True, basic text cleaning is performed.
    :param num_jobs: int, the number of jobs to use for parallel processing.
    :return: A dict with manifests. The keys are: ``{'recordings', 'supervisions'}``.
    """
    #audio_dir = Path(audio_dir)
    #transcript_dir = Path(transcript_dir)
    manifests = {}
    #folders = ["02", "03", "04", "05", "06", "07", ""]
    # for subset in ["dev", "test", "train"]:
    #info(f"Processing Callhome Levantine subset: {subset}")
    # if subset == 'dev':
    #     recordings = RecordingSet.from_dir(Path(audio_dir) / "cts_lev_ara_td5_a_d1/data/00/",
    #                                         pattern='*.sph')
    #     transcript_paths = check_and_rglob(
    #         Path(transcript_dir) / "data/00", "*.txt")
    # elif subset == 'test':
    #     recordings = RecordingSet.from_dir(Path(audio_dir) / "cts_lev_ara_td5_a_d1/data/01/",
    #                                         pattern='*.sph')
    #     transcript_paths = check_and_rglob(
    #         Path(transcript_dir) / "data/01", "*.txt")
    # else:
    # audio_paths = glob(os.path.join(
    # audio_dir, "cts_lev_ara_td5_a_d[123]/data/[01][2-9]/*.sph"))
    audio_paths = glob(os.path.join(
        audio_dir, "*/data/*/*.sph"))
    recordings = RecordingSet.from_recordings(
        Recording.from_file(p) for p in tqdm(audio_paths))

    transcript_paths = glob(os.path.join(
        transcript_dir, "data/*/*.txt"))

    supervisions = []

    for path in transcript_paths:
        idx = 0
        p = Path(path)
        audio_suf = str(p).split('/')[-2]
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line or len(line.split()) < 4:
                continue
            recording_id = (p.stem).lower()
            # example line:
            # 7.29 8.35 A: يعطيك العافية
            start, end, spk, text = line.split(maxsplit=3)
            spk = spk.replace(":", "")
            duration = float(Decimal(end) - Decimal(start))
            if text_cleaning is True:
                text = cleaning(text)
            if duration <= 0 or text == '':
                continue
            start = float(start)

            channel = ord(spk) - ord("A")
            recording_id = f"{recording_id}_{channel}"
            start = float(start)
            # pdb.set_trace()
            supervisions.append(
                SupervisionSegment(
                    id=f"{recording_id}_{spk:0>2s}_{idx:0>5d}",
                    recording_id=recording_id,
                    start=start,
                    duration=duration,
                    speaker=f"{recording_id}_{spk:0>2s}",
                    text=text,
                    channel=channel,
                )
            )
            idx += 1

    supervisions = SupervisionSet.from_segments(supervisions)

    #recordings, supervisions = fix_manifests(recordings, supervisions)
    #validate_recordings_and_supervisions(recordings, supervisions)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        recordings.to_file(
            output_dir / "levant_recordings.jsonl.gz"
        )
        supervisions.to_file(
            output_dir / "levant_supervisions.jsonl.gz"
        )

    manifests['train'] = {"recordings": recordings,
                          "supervisions": supervisions}

    return manifests


words_to_remove = ['%ضجّة', '%أصوات', '%أجنبي', '(())', '%تنفّس', '%مهم', '%تداخل',
                   '%تداخل\\', '%سعال', '%صمت', '%متكلمجديد', '%متكلمجديد\\',
                   '%ضحك']


def remove_special_words(text: str) -> str:
    # remove special tags (non-verbatim) from the transcription text
    for word in words_to_remove:
        if word in text:
            text = text.replace(word, '')
    return text


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


def cleaning(text: str) -> str:
    text = remove_special_words(text)
    text = remove_diacritics(text)
    text = remove_punctuations(text)
    text = remove_non_alphanumeric(text)
    text = remove_extra_space(text)
    text = text.strip()
    return text


if __name__ == "__main__":
    manifests = prepare_callhome_levant(
        '/alt-data/speech/asr_data/LDC/LDC2006S29/',
        '/alt-data/speech/asr_data/LDC/LDC2006T07',
        '/home/local/QCRI/ahussein/data')
