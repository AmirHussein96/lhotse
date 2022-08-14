"""
This data is a Basic Language Resource Kit (BLARK 1.0) for Faroese. It contains 100 
hours of transcribed Faroese speech (over 400 speakers). The BLARK 1.0 for Faroese 
was made by the Project Group Ravnur under the Talutøkni Foundation (https://www.maltokni.fo). 
This project started its work on gathering and creating language resources for Faroese in 
January 2019 and is set to end with the release of BLARK 1.0 in July 2022. The aim was to 
create open-source resources that can be used for language technology for Faroese, while 
the main goal for this project group was to get resources that can be used for Faroese 
automatic speech recognition (ASR). The audio was collected by recording speakers reading texts.
The 433 speakers are aged between 18-83 and divided into the main six dialect areas.
The recordings were made on TASCAM DR-40 Linear PCM audio recorders using the built-in stereo
microphones in WAVE 16 bit with a sample rate of 48kHz. All recordings have been transcribed.
Alongside the recordings and transcriptions there is a pronunciation dictionary complete with PoS-tags,
as well as a text collection of 25 million words and an acoustic model and language model for ASR.
"""

import logging
import shutil
import tarfile
from pathlib import Path
from typing import Dict, Optional, Union
import pdb
import lhotse
from lhotse import (
    Recording,
    RecordingSet,
    SupervisionSegment,
    SupervisionSet,
    validate_recordings_and_supervisions,
)
from lhotse import RecordingSet, Recording, AudioSource
from lhotse.utils import Pathlike, urlretrieve_progress


def download_blark(
    target_dir: Pathlike = ".", force_download: Optional[bool] = False
) -> Path:
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    tar_path = target_dir / "BLARK_1.0_update.tar.gz"
    corpus_dir = target_dir / "BLARK_1.0_update"
    completed_detector = corpus_dir / ".completed"
    if completed_detector.is_file():
        logging.info(
            f"Skipping {tar_path.name} because {completed_detector} exists.")
        return corpus_dir
    if force_download or not tar_path.is_file():
        urlretrieve_progress(
            "http://us.openslr.org/resources/125/BLARK_1.0_update.tar.gz",
            filename=tar_path,
            desc="Downloading BLARK_1.0",
        )
    shutil.rmtree(corpus_dir, ignore_errors=True)
    with tarfile.open(tar_path) as tar:
        tar.extractall(path=target_dir)
    completed_detector.touch()
    return corpus_dir


def get_info(path):
    audio_info = lhotse.audio.info(path)
    return audio_info.samplerate, audio_info.frames, audio_info.duration


def prepare_blark(
    blark_root: Pathlike, output_dir: Optional[Pathlike] = None
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Prepare manifests for the BLARK_1.0 corpus.

    The manifests are created in a dict with three splits: train, dev and test.
    Each split contains a RecordingSet and SupervisionSet in a dict under keys 'recordings' and 'supervisions'.

    :param blark_root: Path to the unpacked BLARK data.
    :return: A dict with standard corpus splits containing the manifests.
    """
    blark_root = Path(blark_root)
    output_dir = Path(output_dir) if output_dir is not None else None
    corpus = {}
    split = "Rdata2"
    root = blark_root / "sound_1.0" / "Rdata2"
    # recordings = RecordingSet.from_recordings(
    #     Recording.from_file(p) for p in (root).glob("*[0-9]*.flac")
    # )
    recs = []
    for p in (root).glob("*[0-9]*.flac"):
        info = get_info(p)
        recs.append(Recording(
            id=Path(p).stem,
            sources=[AudioSource(type='file', source=str(p), channels=[0])],
            sampling_rate=info[0],
            num_samples=info[1],
            duration=info[2],
            transforms=None,
        ))

    recordings = RecordingSet.from_recordings(recs)

    recordings = recordings.resample(16000)
    txt = list((blark_root / "recording_output_1.0").glob("*[0-9]*.txt"))
    assert len(txt) == len(recordings), (
        f"Mismatch: found {len(recordings)} "
        f"sphere files and {len(txt)} TXT files. "
        f"You might be missing some parts of BLARK..."
    )
    # get recordings duration
    rec_dur = {i.id: i.duration for i in recordings}
    segments = []
    for p in txt:
        with p.open() as f:
            for idx, l in enumerate(f):
                channel = 0
                rec_id = Path(p).stem

                data = l.split('\t')
                start, end, *words = data[0:3]
                start, end = float(start), float(end)
                dur = round(end - start, ndigits=8)
                text = " ".join(words)
                text = text.lower()
                if rec_dur[rec_id] < end or dur <= 0 or text == '':
                    # remove segments with ends larger than audio duration
                    # Ex: KSM21_230821 file duration is 765.45 sec
                    #  while transcription contains:   757.5	861.4	Bygningurin stendur fyri einari umvæling
                    #
                    continue

                segments.append(
                    SupervisionSegment(
                        id=f"{rec_id}-{idx}",
                        recording_id=rec_id,
                        start=start,
                        duration=dur,
                        channel=channel,
                        text=text,
                        language="Faroese",
                        speaker=rec_id.split('_')[0],
                    )
                )

    supervisions = SupervisionSet.from_segments(segments)

    corpus[split] = {"recordings": recordings,
                     "supervisions": supervisions}

    validate_recordings_and_supervisions(**corpus[split])

    if output_dir is not None:
        recordings.to_file(
            output_dir / f"blark_recordings_{split}.jsonl.gz")
        supervisions.to_file(
            output_dir / f"blark_supervisions_{split}.jsonl.gz")

    return corpus


# if __name__ == '__main__':
#     path = '/alt-arabic/speech/amir/streaming/kaldi/egs/faroese/BLARK_1.0'
#     outdir = '/alt-arabic/speech/amir/kaldi/egs/blark/s5/data'
#     prepare_blark(path, outdir)
