import torch
import torch.nn as nn
import numpy as np

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


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


def fit(clf: MLP,
        X: torch.Tensor,
        y: torch.Tensor,
        lr: float = 1e-4,
        batch_size: int = 2048,
        num_epochs: int = 40,
        eval_ratio: float = 0.1,
        early_stop_tol: int = 5,
        seed: int = 42) -> MLP:
    X_train, y_train, X_val, y_val = stratified_train_eval_split(X,
                                                                 y,
                                                                 eval_ratio=eval_ratio,
                                                                 seed=seed)
    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=batch_size,
                              shuffle=True)
    
    optim = torch.optim.Adam(clf.parameters(), lr=lr, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss()  # we don't use the weighted variant as class imbalance is mild

    final_clf, max_macro_f1, tol = None, 0.0, 0
    for epoch in range(1, num_epochs):
        # Train
        nn.Module.train(clf, True)

        for batch_X, batch_y in train_loader:
            optim.zero_grad()
            loss = loss_fn(clf(batch_X), batch_y)
            loss.backward()
            optim.step()

        # Eval
        nn.Module.train(clf, False)
        eval_metrics = eval(y_true=y_val,
                            y_pred=clf.predict(X_val),
                            y_logits=clf(X_val).detach())
        acc, macro_f1 = eval_metrics['accuracy'], eval_metrics['macro_f1']

        print(f"Epoch [{epoch}/{num_epochs}] | Val Accuracy: {acc:.4f} | Val Macro F1: {macro_f1:.4f} | AUC: {eval_metrics['auc']:.4f}")

        if macro_f1 > max_macro_f1:
            max_macro_f1 = macro_f1
            final_clf = clf
            tol = 0
        else:
            tol += 1

        if tol == early_stop_tol:
            print(f"\n\t↳ Early stopping at epoch {epoch} with best macro F1 {max_macro_f1:.4f}\n")
            break

    return final_clf


def eval(y_true: torch.Tensor, y_pred: torch.Tensor, y_logits: torch.Tensor) -> dict:
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()
    y_logits = y_logits.cpu().numpy()

    acc = accuracy_score(y_true_np, y_pred_np)
    macro_f1 = f1_score(y_true_np, y_pred_np, average='macro')
    # ROC-AUC for harmfulness
    auc = group_auc(y_true.cpu().numpy(),
                    y_logits)  # malicious classes

    return {
        'accuracy': acc,
        'macro_f1': macro_f1,
        'auc': auc
    }


def stratified_train_eval_split(X: torch.Tensor,
                                y: torch.Tensor,
                                eval_ratio: 0.1,
                                seed: int = 42) -> tuple:
    device, x_dtype, y_dtype = X.device, X.dtype, y.dtype
    X_np = X.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()

    X_train, X_val, y_train, y_val = train_test_split(
        X_np, y_np,
        test_size=eval_ratio,
        random_state=seed,
        stratify=y_np,
        shuffle=True,
    )

    X_train = torch.from_numpy(X_train).to(device=device, dtype=x_dtype)
    X_val = torch.from_numpy(X_val).to(device=device, dtype=x_dtype)
    y_train = torch.from_numpy(y_train).to(device=device, dtype=y_dtype)
    y_val = torch.from_numpy(y_val).to(device=device, dtype=y_dtype)

    return X_train, y_train, X_val, y_val



def group_auc(y_true: np.ndarray, logits: np.ndarray) -> float:
    # Take probability of positive class
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]
    return roc_auc_score(y_true, probs)

