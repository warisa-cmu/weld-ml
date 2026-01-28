import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from datetime import datetime
import os


class DatasetPT(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.Y.shape[0]

    def __getitem__(self, idx):
        X_out = torch.from_numpy(self.X[idx, :]).float()
        Y_out = torch.from_numpy(self.Y[idx, :]).float()
        return X_out, Y_out


class DataHandlerPT(Dataset):
    def __init__(self, _X, _Y, scalerX, scalerY):
        self._X = _X
        self._Y = _Y
        self.scalerX = scalerX
        self.scalerY = scalerY
        self.X_train = None
        self.X_test = None
        self.X_val = None
        self.Y_train = None
        self.Y_val = None
        self.Y_test = None

    def split_and_scale(self, test_size, random_state, val_size=0):
        _X_train, _X_test, _Y_train, _Y_test = train_test_split(
            self._X, self._Y, test_size=test_size, random_state=random_state
        )

        self.scalerX.fit(_X_train)
        self.scalerY.fit(_Y_train)

        if val_size > 0:
            _X_train, _X_val, _Y_train, _Y_val = train_test_split(
                _X_train,
                _Y_train,
                # For example, if you want 80% train, 10% validation, and 10% test:
                # First, split off the test set (10%):
                # Next, split the remaining 90% into train and validation.
                # Since you want 80% train and 10% validation overall, the validation set should be 10/90 = 0.111 of the remaining data.
                test_size=val_size / (1 - test_size),
                random_state=random_state + 100,  # Just make random_state different.
            )
            self.X_val = self.scalerX.transform(_X_val)
            self.Y_val = self.scalerY.transform(_Y_val)

        self.X_train = self.scalerX.transform(_X_train)
        self.X_test = self.scalerX.transform(_X_test)

        self.Y_train = self.scalerY.transform(_Y_train)
        self.Y_test = self.scalerY.transform(_Y_test)

    # This part is different from SKLearn version
    def get_train(self):
        return DatasetPT(X=self.X_train, Y=self.Y_train)

    def get_val(self):
        if self.X_val is None:
            raise Exception("No validation data")
        return DatasetPT(X=self.X_val, Y=self.Y_val)

    def get_test(self):
        return DatasetPT(X=self.X_test, Y=self.Y_test)


class CheckpointHandler:
    @staticmethod
    def list_saved_files(root="."):
        for root, dirs, files in os.walk(root):
            for file in files:
                filepath = os.path.join(root, file)
                filepath = filepath.replace(
                    "\\", "/"
                )  # replace backslash with forward slash
                if file.endswith("pth") or file.endswith("pt"):
                    print(filepath)

    @staticmethod
    def make_dir(folder_path):
        os.makedirs(folder_path, exist_ok=True)

    @staticmethod
    def get_dt():
        return datetime.now().strftime("%Y-%m-%d_%H-%M")

    @staticmethod
    def save(save_path, model, optimizer=None, epoch=None, val_loss=None):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
        }
        torch.save(checkpoint, save_path)

    @staticmethod
    def load(load_path, model, optimizer=None):
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        val_loss = checkpoint["val_loss"]
        return model, optimizer, epoch, val_loss


class EarlyStopper:
    def __init__(self, patience=10, min_delta=0):
        # patience: Number of epochs to wait for improvement before stopping
        # min_delta: Minimum decrease in validation loss to consider as an improvement
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0  # Counts epochs with no significant improvement
        self.min_val_loss = float("inf")  # Tracks the lowest validation loss so far

    def __call__(self, val_loss):
        # If current validation loss is the best so far, it's an improvement
        if val_loss < self.min_val_loss:
            self.min_val_loss = val_loss  # Update best loss
            self.counter = 0  # Reset counter since we have improvement
            return {"best_loss": True, "early_stop": False}
        else:
            # If loss hasn't improved enough (by at least min_delta), count it
            if val_loss > (self.min_val_loss + self.min_delta):
                self.counter += 1
                # If we've waited too long with no improvement, trigger early stop
                if self.counter >= self.patience:
                    return {"best_loss": False, "early_stop": True}

        # Fallback condition
        # (1) Loss has not improved but should wait more (counter < patience)
        # (2) No significant worsening (val_loss is between "min_val_loss" and "min_val_loss + min_delta")
        return {"best_loss": False, "early_stop": False}
