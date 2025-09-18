import torch
import torch.nn as nn
import numpy as np

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------
# Model definition
# ---------------------------
class MLP(nn.Module):
    def __init__(self,
                 d_in: int,
                 n_cls: int,
                 hidden_size: list[int],
                 dropout_p: float,
                 use_ln: bool):
        super().__init__()
        self.ln_in = nn.LayerNorm(d_in) if use_ln else nn.Identity()
        # Identify the two widest layers
        drop_idx = set(sorted(range(len(hidden_size)),
                              key=lambda i: (-hidden_size[i], i))[:min(2, len(hidden_size))])

        layers = []
        d_prev = d_in
        for i, h in enumerate(hidden_size):
            block = []
            block.append(nn.Linear(d_prev, h))
            block.append(nn.GELU())

            # Apply dropout to the two widest layers
            if i in drop_idx:
                block.append(nn.Dropout(dropout_p))

            # Don't apply layer norm to the final hidden layer
            if i < len(hidden_size) - 1:
                block.append(nn.LayerNorm(h))

            layers.append(nn.Sequential(*block))
            d_prev = h

        self.hidden = nn.ModuleList(layers)
        self.out = nn.Linear(d_prev, n_cls)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln_in(x)
        for block in self.hidden:
            x = block(x)

        return self.out(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = self.forward(x)
            
        return out.argmax(dim=-1)
        
    def fit(self, X: torch.Tensor, y: torch.Tensor, **trainer_args):
        trainer = Trainer(self, **trainer_args)
        trainer.fit(X, y)
        return trainer  # return trainer so user can reuse (evaluate, load, etc.)

    def evaluate(self, X: torch.Tensor, y: torch.Tensor, **trainer_args):
        trainer = Trainer(self, **trainer_args)
        return trainer.evaluate(X, y)


# ---------------------------
# Trainer class
# ---------------------------
class Trainer:
    def __init__(self,
                 model: nn.Module,
                 lr: float = 1e-4,
                 batch_size: int = 2048,
                 num_epochs: int = 40,
                 eval_ratio: float = 0.1,
                 early_stop_tol: int = 5,
                 seed: int = 42,
                 save_path: str = "best_model.pt"):
        self.model = model
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.eval_ratio = eval_ratio
        self.early_stop_tol = early_stop_tol
        self.seed = seed
        self.save_path = save_path

        self.loss_fn = nn.CrossEntropyLoss()
        self.optim = torch.optim.Adam(model.parameters(),
                                      lr=self.lr,
                                      weight_decay=1e-2)

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        X_train, y_train, X_val, y_val = self._stratified_split(X, y)
        train_loader = DataLoader(TensorDataset(X_train, y_train),
                                  batch_size=self.batch_size,
                                  shuffle=True)

        max_macro_f1, tol = 0.0, 0

        for epoch in range(1, self.num_epochs + 1):
            # Training
            self.model.train()
            for batch_X, batch_y in train_loader:
                self.optim.zero_grad()
                loss = self.loss_fn(self.model(batch_X), batch_y)
                loss.backward()
                self.optim.step()

            # Evaluation
            self.model.eval()
            metrics = self.evaluate(X_val, y_val)
            acc, macro_f1 = metrics['accuracy'], metrics['macro_f1']

            print(f"Epoch [{epoch}/{self.num_epochs}] "
                  f"| Val Accuracy: {acc:.4f} "
                  f"| Val Macro F1: {macro_f1:.4f} "
                  f"| AUC: {metrics['auc']:.4f}")

            # Early stopping & saving
            if macro_f1 > max_macro_f1:
                max_macro_f1 = macro_f1
                self.save()
                tol = 0
            else:
                tol += 1

            if tol == self.early_stop_tol:
                print(f"\n\t↳ Early stopping at epoch {epoch} "
                      f"with best macro F1 {max_macro_f1:.4f}\n")
                break

    def evaluate(self, X: torch.Tensor, y: torch.Tensor) -> dict:
        y_pred = self.model.predict(X)
        y_logits = self.model(X).detach()

        y_true_np = y.cpu().numpy()
        y_pred_np = y_pred.cpu().numpy()
        y_logits_np = y_logits.cpu().numpy()

        acc = accuracy_score(y_true_np, y_pred_np)
        macro_f1 = f1_score(y_true_np, y_pred_np, average='macro')
        auc = self._group_auc(y_true_np, y_logits_np)

        return {"accuracy": acc, "macro_f1": macro_f1, "auc": auc}

    def save(self):
        torch.save(self.model.state_dict(), self.save_path)
        print(f"✔ Model saved to {self.save_path}")

    def load(self):
        self.model.load_state_dict(torch.load(self.save_path))
        self.model.eval()
        print(f"✔ Model loaded from {self.save_path}")

    # ---------------------------
    # Helper methods
    # ---------------------------
    def _stratified_split(self, X, y):
        device, x_dtype, y_dtype = X.device, X.dtype, y.dtype
        X_np, y_np = X.cpu().numpy(), y.cpu().numpy()

        X_train, X_val, y_train, y_val = train_test_split(
            X_np, y_np,
            test_size=self.eval_ratio,
            random_state=self.seed,
            stratify=y_np,
            shuffle=True,
        )

        return (torch.from_numpy(X_train).to(device, dtype=x_dtype),
                torch.from_numpy(y_train).to(device, dtype=y_dtype),
                torch.from_numpy(X_val).to(device, dtype=x_dtype),
                torch.from_numpy(y_val).to(device, dtype=y_dtype))

    def _group_auc(self, y_true: np.ndarray, logits: np.ndarray) -> float:
        probs = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]
        return roc_auc_score(y_true, probs)
