import os
import numpy as np
import librosa

#CONFIG
DATA_DIR= "."
N_MFCC= 13
HOP_LENGTH= 160          
WIN_LENGTH= 400          
SAMPLE_RATE= 16000

def parse_filename(fname):

    base= os.path.splitext(fname)[0]
    parts= base.split("_")
    if len(parts)< 3:
        return None
    try:
        dig= int(parts[0])
        spe= int(parts[1].replace("speaker", ""))
        return dig, spe
    except ValueError:
        return None

def extract_mfcc(filepath):

    y, sr= librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
    y= np.append(y[0], y[1:] - 0.97 * y[:-1])

    mfcc= librosa.feature.mfcc(
        y=y, sr=sr,
        n_mfcc=N_MFCC,
        hop_length=HOP_LENGTH,
        n_fft=WIN_LENGTH
    )                                              

    delta=librosa.feature.delta(mfcc)           
    delta2=librosa.feature.delta(mfcc, order=2)  
    feats=np.vstack([mfcc, delta, delta2]).T     
    # CMVN - per utterance normalisation
    feats=(feats-feats.mean(axis=0))/(feats.std(axis=0) + 1e-8)
    return feats                                   


def load_dataset(data_dir=DATA_DIR):
    train_X, train_y = [],[]
    test_X,  test_y  = [],[]
    wav_files = sorted(f for f in os.listdir(data_dir) if f.endswith(".wav"))
    for fname in wav_files:
        parsed = parse_filename(fname)
        if parsed is None:
            continue
        digit, speaker = parsed
        fpath = os.path.join(data_dir, fname)
        try:
            feats = extract_mfcc(fpath)
        except Exception as e:
            print(f"  [WARN] Skipping {fname}: {e}")
            continue
        if speaker == 6:
            test_X.append(feats)
            test_y.append(digit)
        else:
            train_X.append(feats)
            train_y.append(digit)
    print(f"Loaded {len(train_X)} train files, {len(test_X)} test files.")
    return train_X, train_y, test_X, test_y

if __name__ == "__main__":
    train_X, train_y, test_X, test_y = load_dataset()
    print("Train sample shape:", train_X[0].shape) 
    print("Test  sample shape:", test_X[0].shape)
    print("Digits present:", sorted(set(train_y)))
