import torch
import torch.nn as nn
from GestureDataset import make_loaders
from GestureClassifier import GestureLSTMClassifier
from video_processing_pipeline import run_batch_processing

def train_model(
    processed_dir,
    target_folders=("back_button", "forward_button"),
    max_seq_len=60,
    batch_size=16,
    epochs=50,
    lr=1e-3,
    hidden_dim=64,
    num_layers=2,
    dropout=0.3,
    patience=10,
    save_path="best_model.pt",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, feature_dim, num_classes = make_loaders(
        processed_dir, list(target_folders), max_seq_len=max_seq_len, batch_size=batch_size
    )


    print(f"Feature dim: {feature_dim} | Classes: {num_classes} ({target_folders})")

    model = GestureLSTMClassifier(
        input_dim=feature_dim,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    best_val_acc = -1.0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for x, mask, lengths, labels in train_loader:
            x, mask, lengths, labels = x.to(device), mask.to(device), lengths.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(x, mask=mask, lengths=lengths)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_total += x.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total


        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for x, mask, lengths, labels in val_loader:
                x, mask, lengths, labels = x.to(device), mask.to(device), lengths.to(device), labels.to(device)
                logits = model(x, mask=mask, lengths=lengths)
                loss = criterion(logits, labels)

                val_loss += loss.item() * x.size(0)
                val_correct += (logits.argmax(dim=1) == labels).sum().item()
                val_total += x.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total
        scheduler.step(val_acc)

        print(
            f"Epoch {epoch:3d} | train_loss {train_loss:.4f} acc {train_acc:.3f} "
            f"| val_loss {val_loss:.4f} acc {val_acc:.3f} | lr {optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": feature_dim,
                    "num_classes": num_classes,
                    "hidden_dim": hidden_dim,
                    "num_layers": num_layers,
                    "dropout": dropout,
                    "label_map": {name: i for i, name in enumerate(target_folders)},
                    "max_seq_len": max_seq_len,
                    "val_acc": val_acc,
                },
                save_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    print(f"\nBest val accuracy: {best_val_acc:.3f} — saved to {save_path}")
    return model



if __name__== "__main__":
    target_folders = ["back_button", "forward_button"]

    run_batch_processing(
        base_dir="/content/drive/MyDrive/YourFolderName",  # wherever the Drive folder mounts
        target_folders=target_folders,
        output_dir="processed_features",
        feature_method="invariant",
    )

    # 2. Train
    model = train_model(
        processed_dir="processed_features",
        target_folders=target_folders,
        max_seq_len=60,
        batch_size=16,
        epochs=50,
    )




