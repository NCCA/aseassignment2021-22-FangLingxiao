import multiprocessing as mp
import torch
import librosa
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import torch 
import torchaudio
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from tqdm.notebook import tqdm, trange
from tqdm.auto import tqdm
import multiprocessing as mp 
from torch.nn import Sequential

# First, all the audio files need to be transform to mel_spectrogram.
def wavs_to_spectrogram(filepath):
    # Sampling the audio data
    audio, sr = torchaudio.load(filepath)
    
    melspec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=2048,
        hop_length=512,
        f_min=0,
        f_max=11025,
        n_mels=128
    )
    audio_transforms = Sequential(
        melspec,                               # Convert it into Mel spectrogram
        torchaudio.transforms.AmplitudeToDB()  # Execute logarithmic operations
    )

    log_mel_spectrogram = audio_transforms(audio)
    
    # mel_spectrogram = librosa.feature.melspectrogram(
    #     y=audio,
    #     sr=sr,
    #     n_fft=2048,
    #     hop_length=512,
    #     fmin=0.0,
    #     fmax=11025.0,
    #     n_mels=128
    # )  
    
    # log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
    # Convert numpy array into tensor
    tensor_log_mel_spectrogram = torch.FloatTensor(log_mel_spectrogram)
    # Data normalization
    tensor_log_mel_spectrogram  = (tensor_log_mel_spectrogram - torch.mean(tensor_log_mel_spectrogram)) / torch.std(tensor_log_mel_spectrogram)
    
    return tensor_log_mel_spectrogram


# defining global path variables
MODEL_DIR = "./saved_models"
DATASET_PREFIX = "./Data/GZTAN"
DATA_PATH = f"{DATASET_PREFIX}/genres_original/gztan_dataset.csv"


class GTZANDataset(Dataset):
    def __init__(self, filepath): 
        super().__init__()
        self.dataframe = pd.read_csv(filepath)  # Load data from CSV filepath defined earlier into a Pandas dataframe
    
    def __len__(self):
        return len(self.dataframe) # Return size of our dataframe
    
    def __getitem__(self, i):
        return self.dataframe.iloc[i] # Return the `i`th item in our dataframe     

# The collate function
def custom_collate_fn(batch, frm=1000):
    #image_batch_tensor = torch.FloatTensor(len(batch), 128, 1500)
    image_tensors = []
    labels = []

    label_mapping = {"blues": 0, "classical": 1, "country": 2, "disco": 3, "hiphop": 4,
                     "jazz": 5, "metal": 6, "pop": 7, "reggae": 8, "rock": 9}
    
    # image_tensors = [wavs_to_spectrogram(f"{DATASET_PREFIX}/{item.iloc[0]}").unsqueeze(0) for item in batch]
    for item in batch:
        spec = wavs_to_spectrogram(f"{DATASET_PREFIX}/{item.iloc[0]}")

        # Truncate the spectrogram
        if spec.shape[-1] > frm:
            spec = spec[:, :, :frm]
        elif spec.shape[-1] > frm:
            padding = frm - spec.shape[-1]
            spec = F.pad(spec, (0, padding, 0, 0), "constant", 0)

        image_tensors.append(spec)
        labels.append(label_mapping[item.iloc[1].strip()])

    #torch.cat(image_tensors, out=image_batch_tensor) # torch.cat simply concatenates a list of individual tensors (image_tensors) into a single Pytorch tensor (image_batch_tensor)
    image_batch_tensor = torch.stack(image_tensors)
    label_batch_tensor = torch.LongTensor(labels) # use the label list to create a torch tensor of ints
    return (image_batch_tensor, label_batch_tensor)

def load_data(data_path, batch_sz=10, train_val_test_split=[0.6, 0.2, 0.2]):
    # This is a convenience funtion that returns dataset splits of train, val and test according to the fractions specified in the arguments
    assert sum(train_val_test_split) == 1, "Train, val and test fractions should sum to 1!"  # Always a good idea to use static asserts when processing arguments that are passed in by a user!
    dataset = GTZANDataset(data_path)  # Instantiating our previously defined dataset
    
    # This code generates the actual number of items that goes into each split using the user-supplied fractions
    tr_va_te = []
    for frac in train_val_test_split:
        actual_count = frac * len(dataset)
        actual_count = round(actual_count)
        tr_va_te.append(actual_count)
    
    # split dataset into train, val and test
    train_split, val_split, test_split = random_split(dataset, tr_va_te)
    
    # Use Pytorch DataLoader to load each split into memory. It's important to pass in our custom collate function, so it knows how to interpret the 
    # data and load it. num_workers tells the DataLoader how many CPU threads to use so that data can be loaded in parallel, which is faster
    n_cpus = 0 # returns number of CPU cores on this machine
    train_dl = DataLoader(train_split, 
                          batch_size=batch_sz, 
                          shuffle=True, 
                          collate_fn=custom_collate_fn,
                          num_workers=n_cpus)            
    val_dl = DataLoader(val_split, 
                        batch_size=batch_sz, 
                        shuffle=True, 
                        collate_fn=custom_collate_fn,
                        num_workers=n_cpus)
    test_dl = DataLoader(test_split,
                         batch_size=batch_sz,
                         shuffle=False,
                         collate_fn=custom_collate_fn,
                         num_workers=n_cpus)
    return train_dl, val_dl, test_dl
