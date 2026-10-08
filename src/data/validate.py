from medmnist import PathMNIST
from torchvision import transforms
from collections import Counter


def inspect_dataset():
    transform = transforms.ToTensor()

    train_dataset = PathMNIST(
        split="train",
        transform=transform,
        download=False
    )

    print("===== PathMNIST Class Validation =====")
    print(f"Training samples: {len(train_dataset)}")

    labels = []

    for i in range(len(train_dataset)):
        _, label = train_dataset[i]
        labels.append(int(label[0]))

    class_counts = Counter(labels)

    print("\nClass distribution:")

    for class_id in sorted(class_counts):
        print(f"Class {class_id}: {class_counts[class_id]} images")

    print(f"\nNumber of unique classes: {len(class_counts)}")


if __name__ == "__main__":
    inspect_dataset()