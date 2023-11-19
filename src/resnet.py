import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.utils.prune as prune

from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torchvision.models import resnet18

import tqdm

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
)

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


# Count the number of parameters that are 0.0
def count_zeros(model):
    zeros = 0

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            zeros += torch.sum(module.weight == 0.0)

    print(f"Number of parameters that are 0.0: {zeros}")
    print(
        f"Total number of parameters: {sum(p.numel() for p in model.parameters())}")
    print(
        f"Percentage of parameters that are 0.0: {zeros / sum(p.numel() for p in model.parameters()) * 100} %"
    )


def eval_model(model, testloader, device):
    all_labels = []
    all_preds = []

    # Asegúrate de que el modelo esté en modo de evaluación
    model.eval()

    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            # Mueve los resultados a 'cpu' y conviértelos a listas para almacenarlos
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Suponiendo que tienes 'all_labels' y 'all_preds' que son las etiquetas verdaderas y las predicciones de todos los datos de prueba
    precision_macro = precision_score(all_labels, all_preds, average="macro")
    recall_macro = recall_score(all_labels, all_preds, average="macro")
    f1_score_macro = f1_score(all_labels, all_preds, average="macro")

    precision_weighted = precision_score(
        all_labels, all_preds, average="weighted")
    recall_weighted = recall_score(all_labels, all_preds, average="weighted")
    f1_score_weighted = f1_score(all_labels, all_preds, average="weighted")

    accuracy = accuracy_score(all_labels, all_preds)
    balanced_accuracy = balanced_accuracy_score(all_labels, all_preds)

    metrics = {
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_score_macro": f1_score_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_score_weighted": f1_score_weighted,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
    }

    return metrics


# Training
def train_model(model, trainloader, testloader, device):
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        running_loss = 0.0
        for i, data in tqdm.tqdm(
            enumerate(trainloader, 0), total=len(trainloader), leave=False
        ):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch} loss: {running_loss / len(trainloader)}")

        # Test set loss
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()

        print(f"Test accuracy: {100 * correct / total}")


resnet = resnet18(weights="IMAGENET1K_V1")


# Transformación para convertir imágenes PIL a tensores.
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

train_dataset = CIFAR10(root="./data", train=True,
                        download=True, transform=transform)
test_dataset = CIFAR10(root="./data", train=False,
                       download=True, transform=transform)

trainloader = DataLoader(train_dataset, batch_size=256,
                         shuffle=True, num_workers=4)
testloader = DataLoader(test_dataset, batch_size=256,
                        shuffle=False, num_workers=4)

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using {device} device")

# Transfer learning
resnet.fc = nn.Linear(512, 10)

resnet.to(device)

train_model(resnet, trainloader, testloader, device)

print("Results before pruning")
print("======================")
print()

# Measure evaluation time
# Evaluate 10 times to get a better average
evaluation_times = []
metrics = {}
for _ in range(10):
    # Measure evaluation time
    start = time.time()
    metrics = eval_model(resnet, testloader, device)
    end = time.time()

    evaluation_times.append(end - start)

average_time = sum(evaluation_times) / len(evaluation_times)
std_time = sum([(time - average_time)**2 for time in evaluation_times]) / \
    len(evaluation_times)

print("================= Evaluation times =================")
print(f"Average time to evaluate without pruning: {average_time} seconds")
print(f"Standard deviation: {std_time}")
print("====================================================")

print()
print("Metrics")
print("=======")
print()
print(f"Accuracy: {metrics['accuracy']}")
print(f"Balanced accuracy: {metrics['balanced_accuracy']}")
print(f"F1 score macro: {metrics['f1_score_macro']}")
print(f"F1 score weighted: {metrics['f1_score_weighted']}")
print(f"Precision macro: {metrics['precision_macro']}")
print(f"Precision weighted: {metrics['precision_weighted']}")
print(f"Recall macro: {metrics['recall_macro']}")
print(f"Recall weighted: {metrics['recall_weighted']}")
print()

# print(resnet)

count_zeros(resnet)

# Part 2: pruning

# Prune 30% of connections in all 2D-conv layers
for name, module in resnet.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        prune.l1_unstructured(module, name="weight", amount=0.3)

# Print model's state_dict
# print("Model's state_dict:")
# for param_tensor in resnet.state_dict():
#     print(param_tensor, "\t", resnet.state_dict()[param_tensor].size())


# Aply mask
for name, module in resnet.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        prune.remove(module, "weight")


# Evaluate 10 times to get a better average
evaluation_times_p = []
metrics_p = {}
for _ in range(10):
    # Measure evaluation time
    start = time.time()
    metrics_p = eval_model(resnet, testloader, device)
    end = time.time()

    evaluation_times_p.append(end - start)

average_time_p = sum(evaluation_times_p) / len(evaluation_times_p)
std_time_p = sum([(time - average_time_p)**2 for time in evaluation_times_p]) / \
    len(evaluation_times_p)

print("================= Evaluation times =================")
print(f"Average time to evaluate with pruning: {average_time_p} seconds")
print(f"Standard deviation: {std_time_p}")
print("====================================================")

print()
print("Metrics")
print("=======")
print()
print(f"Accuracy: {metrics_p['accuracy']}")
print(f"Balanced accuracy: {metrics_p['balanced_accuracy']}")
print(f"F1 score macro: {metrics_p['f1_score_macro']}")
print(f"F1 score weighted: {metrics_p['f1_score_weighted']}")
print(f"Precision macro: {metrics_p['precision_macro']}")
print(f"Precision weighted: {metrics_p['precision_weighted']}")
print(f"Recall macro: {metrics_p['recall_macro']}")
print(f"Recall weighted: {metrics_p['recall_weighted']}")
print()

# Plot results as boxplots
sns.set_theme(style="whitegrid")

data = {
    "Evaluation time": evaluation_times,
    "Evaluation time after pruning": evaluation_times_p,
}

df = pd.DataFrame(data)
ax = sns.boxplot(data=df)

ax = sns.boxplot(data=df, palette="Set3")
ax.set_title("Evaluation time with and without pruning")
ax.set_ylabel("Time (seconds)")
ax.set_xlabel("")
ax.set_xticklabels(["Without pruning", "With pruning"])

sns.despine(offset=10, trim=True)
sns.set(rc={'figure.figsize': (10, 8)})
sns.set(font_scale=1.5)

plt.show()


# Save pruned model
torch.save(resnet.state_dict(), "models/resnet18_cifar10.pt")
