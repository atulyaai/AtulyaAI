# main.py
from models.model import MultimodalModel
import torch

def main():
    print("Instantiating model...")
    model = MultimodalModel()
    print("Loading tokenizer...")
    model.tokenizer.load("models/tokenizer.json")
    print("Running forward pass...")
    audio = torch.zeros(1, 1, 128, 128)  # Correct shape for audio
    video = torch.zeros(1, 3, 16, 64, 64)  # Correct shape for video
    result = model.forward("hello", audio, video)
    print("Forward result:")
    for k, v in result.items():
        print(f"  {k}: {v}")
    print("Saving model...")
    model.save()
    print("Loading model...")
    model.load()
    print("Model save/load successful.")

if __name__ == '__main__':
    main() 