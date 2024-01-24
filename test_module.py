import unittest
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ExponentialLR
from torchmetrics import Accuracy
from torch.utils.tensorboard import SummaryWriter
from DataLoader import load_data, wavs_to_spectrogram
from ResNet import *
from TrainAndVal import train_part

class TestAudioPreprocessing(unittest.TestCase):
    def test_audio_to_spectrogram(self):
        test_audio_path = 'Data/GZTAN/genres_original/country/country.00008.wav'
        spectrogram = wavs_to_spectrogram(test_audio_path)

        self.assertIsInstance(spectrogram, torch.Tensor)

        self.assertEqual(spectrogram.ndim, 3)
        self.assertEqual(spectrogram.shape[1], 128)

class TestDataLoader(unittest.TestCase):
    def test_data_loading(self):
        train_dl, val_dl, test_dl = load_data("Data/GZTAN/genres_original/gztan_dataset.csv", batch_sz=16)
        self.assertIsNotNone(train_dl)
        self.assertIsNotNone(val_dl)
        self.assertIsNotNone(test_dl)


class TestModelInitialization(unittest.TestCase):
    def test_resnet_initialization(self):
        model = get_resnet()
        self.assertIsNotNone(model)


class TestModelOutputDimension(unittest.TestCase):
    def setUp(self):
        self.DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = ResNet(in_channels=1, resblock=ResBlock, outputs=10).to(self.DEVICE)

    def test_output_dimension(self):
        batch_size = 10
        channels = 1
        height = 128
        width = 128

        dummy_input = torch.randn(batch_size, channels, height, width, device=self.DEVICE)

        with torch.no_grad():
            output = self.model(dummy_input)

        expect_shape = (batch_size, 10)
        self.assertEqual(output.shape, expect_shape)
        

class TestTrainFunction(unittest.TestCase):
    def setUp(self):
         #simplified data
        batch_size = 16
        num_samples = 100
        num_features = 128
        num_classes = 10

        # generate random images and labels
        X = torch.randn(num_samples, 1, num_features, num_features)
        y = torch.randint(0, num_classes,(num_samples,))

        dataset = TensorDataset(X,y)
        self.train_dl = DataLoader(dataset, batch_size=batch_size,shuffle=True)


        self.DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = ResNet(in_channels=1, resblock=ResBlock, outputs=10).to(self.DEVICE)
        self.train_dl = load_data("Data/GZTAN/genres_original/gztan_dataset.csv", batch_sz=16)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr = 0.01)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.epochs = 0
        self.sche = ExponentialLR(self.optimizer,gamma=1)
        self.acc_fn = Accuracy(task="multiclass", num_classes=10).to(self.DEVICE)
        self.writer = SummaryWriter()

    def test_train_step(self):
        try:
            train_part(
                epochs=self.epochs,
                model=self.model,
                train_dl=self.train_dl,
                optimizer=self.optimizer,
                lr_scheduler= self.sche,
                acc_fn=self.acc_fn,
                loss_fn=self.loss_fn,
                writer=self.writer,
                DEVICE=self.DEVICE
                )
        except Exception as e:
            print(f"Exception occurred: {e}")
            raise


if __name__ == '__main__':
    unittest.main()