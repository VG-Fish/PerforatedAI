import os
import time

import torch
import torchvision
from torch import nn
from torchvision import transforms

try:
    import torch_xla
    import torch_xla.core.xla_model as xm

    HAS_XLA = True
except ImportError:
    HAS_XLA = False


# Minimal fixed config (no CLI args)
DATA_ROOT = "./Datasets"
BATCH_SIZE = 32
EPOCHS = 5
LR = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-3
NUM_WORKERS = 8
PRINT_FREQ = 50

TRAINIUM_FAST_MODE = True
TRAINIUM_BF16 = True


def is_neuron_device(device):
    return str(device).startswith("xla")


def sync_if_neuron(device):
    if is_neuron_device(device):
        torch_xla.sync()


def get_device():
    if HAS_XLA:
        if TRAINIUM_FAST_MODE:
            if TRAINIUM_BF16:
                os.environ.setdefault("XLA_USE_BF16", "1")
                os.environ.setdefault("NEURON_RT_STOCHASTIC_ROUNDING_EN", "1")
            os.environ.setdefault("NEURON_NUM_RECENT_MODELS_TO_KEEP", "8")
            os.environ.setdefault("NEURON_FUSE_SOFTMAX", "1")

        print(
            "XLA env: "
            f"XLA_USE_BF16={os.environ.get('XLA_USE_BF16', '')}, "
            f"NEURON_RT_STOCHASTIC_ROUNDING_EN={os.environ.get('NEURON_RT_STOCHASTIC_ROUNDING_EN', '')}, "
            f"NEURON_CC_FLAGS={os.environ.get('NEURON_CC_FLAGS', '')}, "
            f"NEURON_COMPILE_CACHE_URL={os.environ.get('NEURON_COMPILE_CACHE_URL', '')}"
        )
        return xm.xla_device()

    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def evaluate(model, loader, device, criterion):
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, target in loader:
            images = images.to(device)
            target = target.to(device)
            logits = model(images)
            loss = criterion(logits, target)

            batch_size = images.size(0)
            loss_sum += float(loss.item()) * batch_size
            correct += int(logits.argmax(dim=1).eq(target).sum().item())
            total += batch_size

    sync_if_neuron(device)
    avg_loss = loss_sum / max(1, total)
    acc1 = 100.0 * correct / max(1, total)
    return avg_loss, acc1


def main():
    device = get_device()
    print(f"Using device: {device}")

    workers = 0 if is_neuron_device(device) else NUM_WORKERS
    if is_neuron_device(device) and NUM_WORKERS > 0:
        print(f"Neuron mode: forcing workers=0 (was {NUM_WORKERS})")

    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    test_tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    print("Loading Food-101...")
    train_ds = torchvision.datasets.Food101(
        root=DATA_ROOT,
        split="train",
        transform=train_tf,
        download=True,
    )
    test_ds = torchvision.datasets.Food101(
        root=DATA_ROOT,
        split="test",
        transform=test_tf,
        download=True,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=workers,
        drop_last=is_neuron_device(device),
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=workers,
        drop_last=is_neuron_device(device),
    )

    model = torchvision.models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 101)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )

    print("Start training")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        t0 = time.time()
        header = f"Epoch [{epoch + 1}/{EPOCHS}]"

        for i, (images, target) in enumerate(train_loader):
            step_start = time.time()
            images = images.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, target)
            loss.backward()
            if is_neuron_device(device):
                xm.optimizer_step(optimizer, barrier=True)
            else:
                optimizer.step()

            batch_size = images.size(0)
            epoch_loss += float(loss.item()) * batch_size
            epoch_correct += int(logits.argmax(dim=1).eq(target).sum().item())
            epoch_total += batch_size

            if i % PRINT_FREQ == 0:
                dt = time.time() - step_start
                imgs_per_s = batch_size / dt if dt > 0 else 0.0
                print(
                    f"{header} [{i}/{len(train_loader)}] "
                    f"lr: {optimizer.param_groups[0]['lr']:.6f} img/s: {imgs_per_s:.2f}"
                )

        sync_if_neuron(device)
        train_loss = epoch_loss / max(1, epoch_total)
        train_acc1 = 100.0 * epoch_correct / max(1, epoch_total)
        test_loss, test_acc1 = evaluate(model, test_loader, device, criterion)
        dt = time.time() - t0

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} train_acc1={train_acc1:.2f} | "
            f"test_loss={test_loss:.4f} test_acc1={test_acc1:.2f} | "
            f"time={dt:.1f}s"
        )


if __name__ == "__main__":
    main()
