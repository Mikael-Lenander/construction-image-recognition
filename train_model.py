import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.transforms import v2
from torchvision.models import get_model
from torchvision.datasets import ImageFolder, folder
from torchmetrics import Accuracy
from pathlib import Path
import os
from typing import Union, Optional, Callable, Any
from collections import Counter
from argparse import ArgumentParser

DATASETS_PATH = Path(os.getcwd())
LOG_BASEDIR = Path("runs")
MODELS_BASEDIR = Path("models")
EPOCHS = 50

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

AUGMENTATION_AUTO, AUGMENTATION_CUSTOM, AUGMENTATION_NONE = "auto", "custom", "none"
CONVNEXT_TINY, CONVNEXT_SMALL, CONVNEXT_BASE, CONVNEXT_LARGE = "convnext_tiny", "convnext_small", "convnext_base", "convnext_large"
CLASSES = ("gravel", "asphalt", "excavation", "sewer-pipe", "cabels", "geotextile")

class SelectiveImageFolder(ImageFolder):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = folder.default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        classes: Optional[list[str]] = None,
    ):
        self.classes = classes
        super().__init__(root, transform, target_transform, loader, is_valid_file)

    def find_classes(self, directory: Union[str, Path]) -> tuple[list[str], dict[str, int]]:
        classes, class_to_idx = super().find_classes(directory)
        if self.classes is None:
            return classes, class_to_idx
        classes = [c for c in classes if c in self.classes]
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes) if cls_name in self.classes}
        return classes, class_to_idx


def get_transforms(augmentation_type, model_size):
    preprocesses = {
        CONVNEXT_TINY: [
            v2.Resize((236, 236), interpolation=v2.InterpolationMode.BILINEAR),
            v2.CenterCrop(224),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ],
        CONVNEXT_SMALL: [
            v2.Resize((230, 230), interpolation=v2.InterpolationMode.BILINEAR),
            v2.CenterCrop(224),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ],
        CONVNEXT_BASE: [
            v2.Resize((232, 232), interpolation=v2.InterpolationMode.BILINEAR),
            v2.CenterCrop(224),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ],
        CONVNEXT_LARGE: [
            v2.Resize((232, 232), interpolation=v2.InterpolationMode.BILINEAR),
            v2.CenterCrop(224),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ],
    }
    preprocess = preprocesses[model_size]
    augmentations = {
        AUGMENTATION_AUTO: v2.TrivialAugmentWide(),
        AUGMENTATION_CUSTOM: v2.RandomApply(
            nn.ModuleList(
                [
                    v2.RandomHorizontalFlip(p=0.5),
                    v2.RandomApply(
                        nn.ModuleList(
                            [v2.ColorJitter(brightness=(0.3, 1.4), contrast=(0.3, 1.4), saturation=(0.3, 1.4))]
                        ),
                        p=0.3,
                    ),
                    v2.RandomApply(
                        nn.ModuleList(
                            [
                                v2.RandomAffine(
                                    degrees=0,
                                    translate=(0.2, 0.2),
                                    scale=(0.7, 1.0),
                                )
                            ]
                        ),
                        p=0.3,
                    ),
                    v2.RandomApply(nn.ModuleList([v2.RandomRotation(degrees=90)]), p=0.3),
                ]
            ),
            p=0.5,
        ),
    }
    if augmentation_type == AUGMENTATION_NONE:
        return v2.Compose(preprocess)
    return v2.Compose(preprocess + [augmentations[augmentation_type]])


def get_datasets(dataset_name, classes, train_transforms, val_transforms):
    dataset_path = DATASETS_PATH / dataset_name
    class_names = [c for c in classes]
    training_data = SelectiveImageFolder(dataset_path / "train", transform=train_transforms, classes=class_names)
    validation_data = SelectiveImageFolder(dataset_path / "val", transform=val_transforms, classes=class_names)
    test_data = SelectiveImageFolder(dataset_path / "test", transform=val_transforms, classes=class_names)
    return training_data, validation_data, test_data


def get_loaders(training_data, validation_data, test_data, batch_size, use_weighted_sampling=True):
    if use_weighted_sampling:
        train_label_counts = list(Counter(training_data.targets).values())
        train_sampler_weights = 1.0 / torch.tensor(train_label_counts, dtype=torch.float32)
        train_sampler = WeightedRandomSampler(
            train_sampler_weights[training_data.targets], len(training_data), replacement=True
        )
        train_loader = DataLoader(
            training_data,
            batch_size=batch_size,
            num_workers=os.cpu_count(),
            sampler=train_sampler,
        )
    else:
        train_loader = DataLoader(training_data, batch_size=batch_size, num_workers=os.cpu_count(), shuffle=True)

    val_loader = DataLoader(validation_data, batch_size=8, num_workers=os.cpu_count())
    test_loader = DataLoader(test_data, batch_size=8, num_workers=os.cpu_count())
    return train_loader, val_loader, test_loader


def get_convnext_model(model_size, num_classes: int):
    model = get_model(model_size, weights="DEFAULT")

    # Change the last layer to match the number of classes in target dataset
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def freeze_all_but_last_n_layers(model, n):
    num_layers = sum([len(block) for block in model.features])
    layer_idx = 0
    for block in model.features:
        for layer in block:
            if layer_idx < num_layers - n:
                for param in layer.parameters():
                    param.requires_grad_(False)
            else:
                for param in layer.parameters():
                    param.requires_grad_(True)
            layer_idx += 1

    for param in model.classifier.parameters():
        param.requires_grad_(True)


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def create_writer(
    experiment_name: str,
    dataset_name,
    classes,
    model_size,
    augmentation_type,
    num_trainable_layers: int,
    batch_size: int,
    use_weighted_sampling: bool,
    learning_rate: float,
):
    log_dir = os.path.join(
        LOG_BASEDIR,
        experiment_name,
        f"{dataset_name}--{len(classes)}-class--{model_size}--aug-{augmentation_type}--tune-{num_trainable_layers}--bs-{batch_size}--ws-{use_weighted_sampling}--lr-{learning_rate}",
    )
    writer = SummaryWriter(
        log_dir=log_dir,
    )
    return writer


def save_model(
    model: nn.Module,
    dataset_name,
    classes,
    model_size,
    augmentation_type,
    num_trainable_layers: int,
    batch_size: int,
    use_weighted_sampling: bool,
    learning_rate: float,
):
    model_name = f"{dataset_name}--{len(classes)}-class--{model_size}--aug-{augmentation_type}--tune-{num_trainable_layers}--bs-{batch_size}--ws-{use_weighted_sampling}--lr-{learning_rate}.pt"
    MODELS_BASEDIR.mkdir(parents=True, exist_ok=True)
    model_save_path = MODELS_BASEDIR / model_name
    print(f"Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)


def train_step(model, train_loader, loss_fn, accuracy_fn, optimizer):
    model.train()
    train_loss, train_accuracy = 0, 0

    for X, y in train_loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        accuracy = accuracy_fn(y_pred.argmax(dim=1), y)
        train_loss += loss.item()
        train_accuracy += accuracy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(train_loader)
    train_accuracy /= len(train_loader)
    print(f"Train loss: {train_loss:.4f} | Train accuracy: {train_accuracy:.4f}")

    return train_loss, train_accuracy


def validation_step(model, val_loader, loss_fn, accuracy_fn, classes=CLASSES):
    model.eval()
    val_loss, val_accuracy = 0, 0
    class_scores = [0] * len(classes)
    class_sums = [0] * len(classes)

    with torch.inference_mode():
        for X, y in val_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            accuracy = accuracy_fn(y_pred.argmax(dim=1), y)
            val_loss += loss
            val_accuracy += accuracy
            for i in range(len(classes)):
                class_mask = y == i
                class_scores[i] += torch.sum(y_pred[class_mask].argmax(dim=1) == y[class_mask]).item()
                class_sums[i] += len(y[class_mask])

        val_loss /= len(val_loader)
        val_accuracy /= len(val_loader)
        class_accuracy = [score / sum for score, sum in zip(class_scores, class_sums)]
        print(f"Validation loss: {val_loss:.4f} | Validation accuracy: {val_accuracy:.4f}")
        print(f"Class validation accuracies: {', '.join([f'{c}: {a:.4f}' for c, a in zip(classes, class_accuracy)])}")

    return val_loss, val_accuracy, class_accuracy


def train(
    dataset_name,
    classes,
    model_size,
    augmentation_type,
    num_trainable_layers: int,
    batch_size: int,
    use_weighted_sampling: bool,
    learning_rate: float,
    experiment_name: str,
    use_writer=False,
):
    if use_writer:
        writer = create_writer(
            experiment_name,
            dataset_name,
            classes,
            model_size,
            augmentation_type,
            num_trainable_layers,
            batch_size,
            use_weighted_sampling,
            learning_rate,
        )
        print(f"Training {writer.log_dir}")
        print("-------------------------------")
    train_transform = get_transforms(augmentation_type, model_size)
    val_transform = get_transforms(AUGMENTATION_NONE, model_size)
    training_data, validation_data, test_data = get_datasets(dataset_name, classes, train_transform, val_transform)
    train_loader, val_loader, test_loader = get_loaders(
        training_data, validation_data, test_data, batch_size, use_weighted_sampling
    )
    model = get_convnext_model(model_size, len(classes))
    freeze_all_but_last_n_layers(model, num_trainable_layers)

    accuracy_fn = Accuracy(task="multiclass", num_classes=len(classes))
    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    early_stopper = EarlyStopper(patience=10, min_delta=0.01)

    model.to(DEVICE)
    accuracy_fn.to(DEVICE)

    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch}\n-------------------------------")
        train_loss, train_accuracy = train_step(model, train_loader, loss_fn, accuracy_fn, optimizer)
        val_loss, val_accuracy, class_accuracy = validation_step(model, val_loader, loss_fn, accuracy_fn)

        if writer:
            writer.add_scalars("Loss", {"train": train_loss, "validation": val_loss}, epoch)
            writer.add_scalars("Accuracy", {"train": train_accuracy, "validation": val_accuracy}, epoch)
            writer.add_scalars("Class_accuracy", {c: a for c, a in zip(classes, class_accuracy)}, epoch)
            writer.add_graph(model, input_to_model=torch.randn(batch_size, 3, 224, 224).to(DEVICE))

        if early_stopper.early_stop(val_loss):
            print("Early stopping")
            break

    if writer:
        writer.flush()
        writer.close()

    save_model(
        model,
        dataset_name,
        classes,
        model_size,
        augmentation_type,
        num_trainable_layers,
        batch_size,
        use_weighted_sampling,
        learning_rate,
    )


def extract_params():
    parser = ArgumentParser()
    parser.add_argument("--experiment-name", type=str, required=True)
    parser.add_argument("--model", type=str, choices=(CONVNEXT_TINY, CONVNEXT_SMALL, CONVNEXT_BASE, CONVNEXT_LARGE), required=True)
    parser.add_argument("--augmentation-type", type=str, choices=(AUGMENTATION_AUTO, AUGMENTATION_CUSTOM, AUGMENTATION_NONE), required=True)
    parser.add_argument("--num-trainable-layers", type=int, choices=range(0, 50), required=True)
    parser.add_argument("--use-weighted-sampling", type=bool, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = extract_params()
    train(
        dataset_name='dataset',
        classes=CLASSES,
        model_size=args.model,
        augmentation_type=args.augmentation_type,
        num_trainable_layers=args.num_trainable_layers,
        batch_size=32,
        use_weighted_sampling=args.use_weighted_sampling,
        learning_rate=args.learning_rate,
        experiment_name=args.experiment_name,
        use_writer=True,
    )
