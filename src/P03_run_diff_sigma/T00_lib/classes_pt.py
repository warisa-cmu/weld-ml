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
