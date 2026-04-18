import torch

# data = {"prompt": torch.tensor([1]), "images": torch.tensor([2])}
# torch.save(data, "data.pt")
loaded_data = torch.load("data.pt")
print(loaded_data)

