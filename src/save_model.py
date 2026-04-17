import torch

# Save model
torch.save(model.state_dict(), "models/bert_model.pt")

print("✅ Model saved!")