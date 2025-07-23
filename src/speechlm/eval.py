import torch
import torch.nn.functional as F


@torch.inference_mode()
def _eval(
    model,
    loader: torch.utils.data.DataLoader,
    out_file,
):
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        for batch in loader:
            # Speech LM
            input_ids = batch["input_ids"].to(model.device)
            labels = batch["labels"].to(model.device)
            logits = model(input_ids=input_ids, labels=labels).logits.transpose(1, 2)

            labels = F.pad(labels, (0, 1), value=-100)
            shifted_labels = labels[:, 1:]

            scores = -F.cross_entropy(logits, shifted_labels, reduction="none")
            scores = scores.sum(dim=1) / scores.ne(0).sum(dim=1)
            scores = scores.tolist()

            for id, score in zip(batch["id"], scores):
                f.write(f"{id} {score}\n")
