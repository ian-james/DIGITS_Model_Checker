from datasets import load_dataset

#dataset = load_dataset("Hands900")

# # Load the dataset training
dataset = load_dataset("hands-2.7k", split="train")
dataset
