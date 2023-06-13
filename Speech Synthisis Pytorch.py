import torch
import lws
import scipy
import numpy as np
from scipy import signal
from deepvoice3_pytorch import frontend
from pydub import AudioSegment
import os







#Pytorch Synthesis Wrapper
def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def load_checkpoint(path, model, optimizer, reset_optimizer):
    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    model.load_state_dict(checkpoint["state_dict"])
    return model
def inv_spectrogram(spectrogram):
    S = _db_to_amp(_denormalize(spectrogram) + ref_level_db)  # Convert back to linear
    processor = _lws_processor()
    D = processor.run_lws(S.astype(np.float64).T ** power_)
    y = processor.istft(D).astype(np.float32)
    return inv_preemphasis(y)

def _inv_preemphasis(x, coef=0.97):
    b = np.array([1.0], x.dtype)
    a = np.array([1.0, -coef], x.dtype)
    return signal.lfilter(b, a, x)

def inv_preemphasis(x):
    return _inv_preemphasis(x, preemphasis)
  
def _denormalize(S):
    return (np.clip(S, 0, 1) * -min_level_db) + min_level_db

def _lws_processor():
    return lws.lws(fft_size, hop_size, mode="speech")

def _db_to_amp(x):
    return np.power(10.0, x * 0.05)

def _tts(model, text, p=0, speaker_id=0, fast=False):
    model = model.to(device)
    model.eval()
    if fast:
        model.make_generation_fast_()

    sequence = np.array(_frontend.text_to_sequence(text, p=p))
    sequence = torch.from_numpy(sequence).unsqueeze(0).long().to(device)
    text_positions = torch.arange(1, sequence.size(-1) + 1).unsqueeze(0).long().to(device)
    speaker_ids = None if speaker_id is None else torch.LongTensor([speaker_id]).to(device)

    # Greedy decoding
    with torch.no_grad():
        mel_outputs, linear_outputs, alignments, done = model(
            sequence, text_positions=text_positions, speaker_ids=speaker_ids)

    linear_output = linear_outputs[0].cpu().data.numpy()

    # Predicted audio signal
    waveform = inv_spectrogram(linear_output.T)
    return waveform

def tts(model, text, file_path, p=0, speaker_id=0, fast=True, figures=False):
    waveform = _tts(model, text, p, speaker_id, fast)
    scipy.io.wavfile.write(file_path, rate=fs, data=waveform)




#Hparams
preemphasis=0.97
min_level_db = -100
ref_level_db = 20
power_ = 1.4
fft_size = 1024
hop_size = 256
fs = 22050
speakerid = 90

use_cuda = torch.cuda.is_available()
_frontend = getattr(frontend, 'en')
device = torch.device("cuda" if use_cuda else "cpu")

checkpoint_path = "20171222_deepvoice3_vctk108_checkpoint_step000300000.pth"

model_ = torch.load("Test-2.pth")
model = load_checkpoint(checkpoint_path, model_, None, True)







def SplitChar(word): 
    return [char for char in word]

def sent_tokens(text):
    text = str(text)
    text = text.replace(". ...", ".").replace("...", ".").replace(". ..", ".").replace("∞", "infinity")
    if len(SplitChar(text)) < 100:
        return[text]

    n = 100
    split_ = [text[i:i+n] for i in range(0, len(text), n)]
    p = []
    for i in split_[:-1]:
        x = i[-50:].split(" ")[-3:-1] #20
        p.append(i.replace(" ".join(x), " ".join(x) + "?????", 1))

    p.append(split_[-1])
    return ''.join(p).split("?????")



def removeConcatFiles():
    try:
        for filename in os.listdir("Concatenate/"):
            os.remove(f"Concatenate/{str(filename)}")
    except OSError:
        pass



def ConcatenateAudiofiles():
    combined = AudioSegment.from_file("Concatenate/sound0.wav", format="wav")
    for filename in sorted(os.listdir("Concatenate/")):
        if str(filename) != "sound0.wav":
            combined += AudioSegment.from_file(f"Concatenate/{str(filename)}", format="wav")

    combined.export("combined.wav", format="wav")






answer = "Good luck with your endevours, and I wish you the best, I know you will do great!"


# concatenate audio files 
Answer_sentences = sent_tokens(str(answer))
removeConcatFiles()
for number, sentence in  enumerate(Answer_sentences):
    NNFileName = str(f"Concatenate/sound{number}.wav")
    tts(model, sentence, NNFileName, speaker_id=speakerid, figures=False)

ConcatenateAudiofiles()










