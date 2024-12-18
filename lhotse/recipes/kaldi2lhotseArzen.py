import os
from pathlib import Path
from lhotse import Recording, RecordingSet, SupervisionSegment, SupervisionSet
# from lhotse.audio import AudioSource
from lhotse import AudioSource, validate_recordings_and_supervisions
import soundfile as sf
from lhotse.qa import (
    remove_missing_recordings_and_supervisions,
    trim_supervisions_to_recordings,
)
import re
import sys
from camel_tools.utils.dediac import dediac_ar
from camel_tools.utils.normalize import normalize_alef_ar
from camel_tools.utils.normalize import normalize_alef_maksura_ar

def remove_annotations(line):
    """
    Cleans annotations like (LAUGHTER), special symbols, and reduces multiple spaces.
    """
    line = re.sub(r"(\[|\()\s*(LAUGHTER|LAUGH|COUGH|NOISE|HES|HUM|BREATH)\s*(\]|\))", " ", line, flags=re.IGNORECASE)
    line = re.sub(r'%#?\w+', " ", line)
    line = re.sub(r"(//|=|#|\(\(|\)\)|@|\$)", " ", line)
    if "c++" not in line:
        line = re.sub(r"(\+|\*|\#)", " ", line)
    else:
        line = re.sub(r"(\+|\*|\#)", " ", line)
        line = line.replace("ب c   language", "ب c++ language")
    line = re.sub('\s{2,}', ' ', line)
    line = line.replace("…", "..").strip()
    return line

def removePunctuation(transcription):
    """
    Removes Arabic and English punctuation marks.
    """
    for symbol in ['.', '،', ',', '?', ':', '؟', '!', ':', ";", "/", "؛"]:
        transcription = transcription.replace(symbol, " ")
    transcription = re.sub(' +', ' ', transcription)
    return transcription

def apply_arabic_processing(transcription):
    """
    Performs Arabic Alef normalization and removes diacritics.
    """
    return dediac_ar(normalize_alef_ar(transcription))

def read_kaldi_files(wav_scp, segments, text, utt2spk):
    """Read Kaldi-like files into dictionaries."""
    wavs = {}
    with open(wav_scp, "r") as f:
        for line in f:
            reco_id, path = line.strip().split(maxsplit=1)
            wavs[reco_id] = path

    utt_to_segments = {}
    with open(segments, "r") as f:
        for line in f:
            utt_id, reco_id, start, end = line.strip().split()
            utt_to_segments[utt_id] = {
                "recording_id": reco_id,
                "start": float(start),
                "end": float(end),
            }

    utt_to_text = {}
    with open(text, "r") as f:
        for line in f:
            utt_id, transcript = line.strip().split(maxsplit=1)
            utt_to_text[utt_id] = transcript

    utt_to_speaker = {}
    with open(utt2spk, "r") as f:
        for line in f:
            utt_id, speaker_id = line.strip().split()
            utt_to_speaker[utt_id] = speaker_id

    return wavs, utt_to_segments, utt_to_text, utt_to_speaker

def create_recordings(wavs):
    """Create RecordingSet from wav.scp."""
    recordings = []
    for reco_id, path in wavs.items():
        # Assuming mono-channel and 16kHz audio
        audio_sf = sf.SoundFile(str(path))
        recording = Recording(
            id=reco_id,
            sources=[AudioSource(type="file", channels=[0], source=path)],
            sampling_rate=16000,
            num_samples=audio_sf.frames,
            duration=audio_sf.frames / audio_sf.samplerate,
        )
        recordings.append(recording)
    return RecordingSet.from_recordings(recordings)

def create_supervisions(utt_to_segments, utt_to_text, utt_to_speaker):
    """Create SupervisionSet from segments, text, and utt2spk."""
    supervisions = []
    for utt_id, segment_info in utt_to_segments.items():
        recording_id = segment_info["recording_id"]
        start = segment_info["start"]
        end = segment_info["end"]
        duration = end - start
        transcript = utt_to_text.get(utt_id, "")
        transcript = apply_arabic_processing(
            removePunctuation(
                remove_annotations(transcript)
            ).lower()
        )
        speaker = utt_to_speaker.get(utt_id, "")
        # Add LID and delimiter
        lid, transcript_with_lid, cs = add_lid(transcript)
        if utt_id == "S04-C03-R04_011631-012393":
            breakpoint()
        supervision = SupervisionSegment(
            id=utt_id,
            recording_id=recording_id,
            start=start,
            duration=duration,
            channel=0,
            text=transcript_with_lid,
            speaker=speaker,
            custom={"lid": lid, "cs": cs},
        )
        supervisions.append(supervision)
    return SupervisionSet.from_segments(supervisions)

def generate_manifests(kaldi_dir, split_dir, output_dir):
    """Generate Lhotse manifests from Kaldi-like files."""
    wav_scp = Path(kaldi_dir) / "wav.scp"
    segments = Path(kaldi_dir) / "segments"
    text = Path(kaldi_dir) / "text"
    utt2spk = Path(kaldi_dir) / "utt2spk"

    # Read Kaldi files
    wavs, utt_to_segments, utt_to_text, utt_to_speaker = read_kaldi_files(
        wav_scp, segments, text, utt2spk
    )

    # Create Lhotse RecordingSet and SupervisionSet
    recordings = create_recordings(wavs)

    supervisions = create_supervisions(utt_to_segments, utt_to_text, utt_to_speaker)

    # Read split files
    split_files = {split: set(Path(split_dir, split).read_text().strip().splitlines()) for split in ("train", "dev", "test")}
    # Validate and save
    recordings, supervisions = remove_missing_recordings_and_supervisions(
            recordings, supervisions,
        )
    manifests = {}
    for split in ("train", "dev", "test"):
        sups_ = supervisions.filter(lambda s: s.recording_id in split_files[split])
        recs_ = recordings.filter(lambda r: r.id in split_files[split])

        # Validate recordings and supervisions
        recs_, sups_ = remove_missing_recordings_and_supervisions(recs_, sups_)
        sups_ = trim_supervisions_to_recordings(recs_, sups_)
        validate_recordings_and_supervisions(recs_, sups_)

        manifests[split] = {"recordings": recs_, "supervisions": sups_}

        # Save to files
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            recs_.to_file(output_dir / f"arzen_recordings_{split}.jsonl.gz")
            sups_.to_file(output_dir / f"arzen_supervisions_{split}.jsonl.gz")
            print(f"Saved {split} split to {output_dir}")

    return manifests

def get_lid(word: str) -> str:
    """
    Determines the language ID (LID) for each word based on whether it contains Arabic characters.
    """
    # Check if the word contains Arabic characters
    if re.search('[\u0600-\u06FF]', word):
        return "<ar>"  # Arabic language ID
    else:
        return "<en>"  # Assume English for non-Arabic characters

# def add_lid(txt, delimiter="|"):
#     """
#     Splits the text into Arabic and English segments and inserts delimiters.
#     Also returns the language labels for each word.
#     Args:
#     txt (str): The input text to be processed.
#     delimiter (str): The delimiter to insert between code-switched segments.
    
#     Returns:
#     lid (list): List of language IDs for each word.
#     new_txt (str): The text with delimiters inserted between language switches.
#     cs (bool): A flag indicating if there was any code-switching.
#     """
#     # Split the input text into words (basic tokenization)
#     txt = txt.split()
#     new_txt = []
#     lid = []
#     prev = ""
#     cs = False  # Flag to track code-switching
    
#     for i, word in enumerate(txt):
#         curr_lid = get_lid(word)  # Get language ID for the current word
        
#         # Check for code-switching (switch between Arabic and English)
#         if curr_lid != prev:
#             if i > 0:
#                 new_txt.append(delimiter)  # Insert delimiter when code-switching
#                 cs = True  # Mark that code-switching occurred
#             new_txt.append(word)
#             lid.append(curr_lid)
#             prev = curr_lid  # Update previous language ID
#         new_txt.append(word)
    
#     return lid, " ".join(new_txt), cs


def add_lid(txt, delimiter="|"):
    """
    Splits the text into Arabic and English segments and inserts delimiters.
    Also returns the language labels for each word.
    Args:
    txt (str): The input text to be processed.
    delimiter (str): The delimiter to insert between code-switched segments.
    
    Returns:
    lid (list): List of language IDs for each word.
    new_txt (str): The text with delimiters inserted between language switches.
    cs (bool): A flag indicating if there was any code-switching.
    """
    # Split the input text into words (basic tokenization)
    txt = txt.split()
    new_txt = []
    lid = []
    prev = ""
    cs = False  # Flag to track code-switching
    
    for i, word in enumerate(txt):
        curr_lid = get_lid(word)  # Get language ID for the current word
        
        # Check for code-switching (switch between Arabic and English)
        if curr_lid != prev:
            if i > 0:
                new_txt.append(delimiter)  # Insert delimiter when code-switching
                cs = True  # Mark that code-switching occurred
            lid.append(curr_lid)
            prev = curr_lid  # Update previous language ID
        
        new_txt.append(word)  # Append the word to the new text
    
    return lid, " ".join(new_txt), cs




if __name__ == "__main__":
    kaldi_dir = "/eph/nvme0/codeswitching/codeswitching/ArzEn_SpeechCorpus_1.0/ASR_files"  # Update with the path to your Kaldi files
    output_dir = "/eph/nvme0/ahussein/icefall/egs/Arzen/ASR/data/manifests"  # Update with desired output 
    split_dir = "/eph/nvme0/codeswitching/codeswitching/ArzEn_SpeechCorpus_1.0/splits"  # Path to split files
    generate_manifests(kaldi_dir, split_dir, output_dir)
