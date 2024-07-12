import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

dataset = load_dataset("google/docci", trust_remote_code=True)

clip_model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32", device_map="auto"
)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def collate_fn(batch):
    return clip_processor(images=[b["image"] for b in batch], return_tensors="pt")


dataloader = DataLoader(
    dataset["train"],
    batch_size=128,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=16,
    pin_memory=True,
)
features_tensors = []

for batch in tqdm(dataloader):
    batch_device = {k: v.to(clip_model.device) for k, v in batch.items()}
    with torch.no_grad():
        image_features = clip_model.get_image_features(**batch_device)
        features_tensors.append(image_features)

# stack all features
out = torch.concat([f for f in features_tensors], dim=0)
out = out.cpu()
# normalize the image features
out = F.normalize(out, p=2, dim=1)
torch.save(out, "clip_features.pt")
sim_matrix = out @ out.T
# mask diagonal
mask = torch.eye(sim_matrix.shape[0], dtype=bool)
sim_matrix[mask] = 0
# save the similarity matrix
torch.save(sim_matrix, "sim_matrix.pt")
