from medmnist import PathMNIST
from torchvision import transforms


def create_datasets():
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = PathMNIST(
        split="train",
        transform=transform,
        download=False
    )

    val_dataset = PathMNIST(
        split="val",
        transform=transform,
        download=False
    )

    test_dataset = PathMNIST(
        split="test",
        transform=transform,
        download=False
    )

    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    train_dataset, val_dataset, test_dataset = create_datasets()

    print("===== PathMNIST Preprocessing Pipeline =====")
    print(f"Training samples:   {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples:       {len(test_dataset)}")

    image, label = train_dataset[0]

    print(f"Image shape: {image.shape}")
    print(f"Label: {label}")
    print(f"Image dtype: {image.dtype}")
    print(f"Pixel range: {image.min().item():.4f} - {image.max().item():.4f}")