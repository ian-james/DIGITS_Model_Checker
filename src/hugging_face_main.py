# from datasets import load_dataset

# dataset = load_dataset("Samsung/samsum")

from huggingface_hub import Repository

from datasets import load_dataset_builder
from datasets import load_dataset
from datasets import get_dataset_split_names
from datasets import get_dataset_config_names
from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset

# ds_builder = load_dataset_builder("rotten_tomatoes")

# ds_builder.info.description

# ds_builder.info.features


# # Load the dataset
# dataset = load_dataset("rotten_tomatoes", split="train")


# # Split the dataset
# get_dataset_split_names("rotten_tomatoes")
# ['train', 'validation', 'test']

# # Load the dataset training
# dataset = load_dataset("rotten_tomatoes", split="train")
# dataset

# # Get configuration if needed.
# configs = get_dataset_config_names("PolyAI/minds14")
# print(configs)


# Remote Code
# c4 = load_dataset("c4", "en", split="train", trust_remote_code=True)
# get_dataset_config_names("c4", trust_remote_code=True)
# get_dataset_split_names("c4", "en", trust_remote_code=True)

from datasets import load_dataset, Image
from torch.utils.data import DataLoader
import torch

dataset = load_dataset("beans", split="train")
dataset = dataset.cast_column("image", Image(mode="RGB"))

from torchvision.transforms import Compose, ColorJitter, ToTensor

jitter = Compose(
    [ColorJitter(brightness=0.5, hue=0.5), ToTensor()]
)

def transforms(examples):
    examples["pixel_values"] = [jitter(image.convert("RGB")) for image in examples["image"]]
    return examples

dataset = dataset.with_transform(transforms)

from torch.utils.data import DataLoader

def collate_fn(examples):
    images = []
    labels = []
    for example in examples:
        images.append((example["pixel_values"]))
        labels.append(example["labels"])
        
    pixel_values = torch.stack(images)
    labels = torch.tensor(labels)
    return {"pixel_values": pixel_values, "labels": labels}
dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=4)