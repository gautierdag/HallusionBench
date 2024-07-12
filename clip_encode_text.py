import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
import pandas as pd

for model_name in ["minicpm", "haiku", "llava", "bunny"]:
    statements_df = pd.read_csv(f"evaluated_statements_{model_name}.csv")

    clip_model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32", device_map="auto"
    )
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def collate_fn(batch):
        return (clip_processor(text=batch, return_tensors="pt", padding=True), batch)

    dataloader = DataLoader(
        list(statements_df.statement.unique()),
        batch_size=16,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=16,
        pin_memory=True,
    )
    features_tensors = {}
    for batch, texts in tqdm(dataloader):
        batch_device = {k: v.to(clip_model.device) for k, v in batch.items()}
        with torch.no_grad():
            text_features = clip_model.get_text_features(**batch_device)
            for text, text_features in zip(texts, text_features):
                features_tensors[text] = F.normalize(text_features, p=2, dim=0).cpu()

    import pickle

    pickle.dump(features_tensors, open(f"clip_features_{model_name}.pkl", "wb"))
    # stack all features
    # out = torch.concat([f for f in features_tensors], dim=0)
    # out = out.cpu()
    # normalize the image features
    # out = F.normalize(out, p=2, dim=1)
    # torch.save(out, f"clip_features_{model_name}.pt")
