from transformer_lens import HookedTransformer
from datasets import load_dataset
from tqdm import tqdm
import torch
import os

os.environ["HF_TOKEN"] = "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"

# download models
gemma_2b_it = HookedTransformer.from_pretrained("gemma-2-2b-it")

# prep data
dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
dataset_iter = iter(dataset)
first_500_points = [next(dataset_iter)["text"] for _ in range(500)]

del dataset, dataset_iter
tokens = gemma_2b_it.to_tokens(
    first_500_points, prepend_bos=True, padding_side="left", move_to_device=True
).detach()
del first_500_points

# prepare hooks
gemma_2b_activations = []
gemma_2b_it_activations = []


def gemma_2b_hook(activations, hook):
    act = activations[..., -1, :].detach().cpu()
    gemma_2b_activations.append(act)


def gemma_2b_it_hook(activations, hook):
    act = activations[..., -1, :].detach().cpu()
    gemma_2b_it_activations.append(act)


# Run inference
# I split this into smaller chunks for memory

# print used memory
print(torch.cuda.memory_summary(device=None, abbreviated=False))

BATCH_SIZE = 1
for i in tqdm(range(0, len(tokens), BATCH_SIZE)):
    ans = gemma_2b_it.run_with_hooks(
        tokens[i : i + BATCH_SIZE],
        fwd_hooks=[("blocks.12.hook_resid_post", gemma_2b_it_hook)],
    )
    del ans
    torch.cuda.empty_cache()


del gemma_2b_it
gemma_2b = HookedTransformer.from_pretrained("gemma-2-2b")


for i in tqdm(range(0, len(tokens), BATCH_SIZE)):
    ans = gemma_2b.run_with_hooks(
        tokens[i : i + BATCH_SIZE],
        fwd_hooks=[("blocks.12.hook_resid_post", gemma_2b_hook)],
    )
    del ans
    torch.cuda.empty_cache()

# concatenate activations
gemma_2b_activations = torch.cat(gemma_2b_activations, dim=0)
gemma_2b_it_activations = torch.cat(gemma_2b_it_activations, dim=0)

del tokens, gemma_2b

torch.save(gemma_2b_activations.detach().cpu(), "/workspace/gemma_2b_activations.pt")
torch.save(
    gemma_2b_it_activations.detach().cpu(), "/workspace/gemma_2b_it_activations.pt"
)
