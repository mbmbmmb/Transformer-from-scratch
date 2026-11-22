# Learning notes

My working notes from building this project, lightly edited: formatting fixed, local paths and
session-management notes removed, old file names updated to the current module names. The content
is what I wrote down at the time (mid-September 2026, after the first training runs). Results that
came later are in [EXPERIMENTS.md](EXPERIMENTS.md).

## Learning path

1. Understand and validate BPE training.
2. Implement and debug BPE tokenizer encode/decode.
3. Work through small tensor operations: linear, embedding, SiLU, softmax, RMSNorm, cross entropy.
4. Build attention pieces: scaled dot-product attention, causal masking, multi-head reshaping, RoPE.
5. Assemble the Transformer block and the full LM.
6. Add training utilities: batching, AdamW, LR schedule, gradient clipping, checkpoints.
7. Train on TinyStories, generate text, then run the experiments and ablations.

## Model bugs found and fixed

- `transformer_lm.py`: the blocks were held in a plain `self.transform_layers = []`. `nn.Module.__setattr__`
  can't see into a Python list, so the layers were never registered. `.to()` skipped them (this
  surfaced as `mps:0 and cpu`), `model.parameters()` left them out (the optimizer was not training
  them), and `state_dict()` would have saved an incomplete model. Fixed with `nn.ModuleList()`.
  Lesson: a loud device error can hide a silent CPU-only bug. `ModuleList` supports
  append/index/len/iterate; it lacks `remove`, `sort`, `index`, `count`, `clear`.
- `transformer_lm.py`: the LM-head `Linear` had `in_features`/`out_features` swapped. The tests never
  caught it because they always construct the model with `weights=`, so the passed tensor's shape wins
  and both arguments are dead. **Gap to remember: fixtures that supply weights never exercise the init path.**
- `masked_fill_` → `masked_fill`. The trailing `_` means in-place. In-place ops on tensors that autograd
  needs for backward can raise, and they clobber shared or broadcast inputs.
- The causal mask is built directly as bool,
  `torch.triu(torch.ones(seq, seq, dtype=torch.bool, device=x.device), diagonal=1)`.
  The earlier `ones_like(scale_dot)` + triu + cast allocated a full `[batch, heads, seq, seq]` float
  tensor per forward just to throw it away.

## Device rules

- Params and buffers follow `model.to(device)`, called once at setup. Never call `.to(x.device)` in `forward`.
- Tensors created inside `forward` (masks, `arange`) must take `device=x.device`, because nothing else moves them.
- Never cache `self.device`; it goes stale after `.to()`. Derive it instead: `next(self.parameters()).device`.
- `rope.py` is the reference pattern: `register_buffer` makes `cos`/`sin` move with `.to()`.

## Training script (`train.py`) structure

- Run it as `python -m cs336_basics.train` from the repo root. Running the file by path breaks:
  `sys.path[0]` becomes `cs336_basics/` and the package can't be imported. A
  `sys.path.append("cs336_basics")` hack makes the imports *resolve*, but it loads modules under a
  different identity, so their own relative imports fail one level deeper.
- The module level only defines things (imports, `REPO_ROOT`, `def`s); the `__main__` guard only invokes.
- `REPO_ROOT = Path(__file__).resolve().parent.parent`. Resolve every configured path in `load_config`,
  so `main` gets absolute values and runs from any cwd.
- Config sections that are `**`-unpacked must match a function signature exactly
  (`model`, `optim`, `data`, `lr_schedule`, `gradient_clipping`, `tokenizer`); `training`/`logging`
  are indexed by hand. Adding a key to a kwargs section breaks the call.
- CLI overrides: `partition("=")`, `split(".")`, walk the dict, coerce the value. Don't use
  `yaml.safe_load` for that: it returns `'1e-4'` as a *string*. Guard unknown keys with
  `if leaf not in d: raise`. YAML 1.1 also parses a bare `1e-3` as a string, so `1.0e-3` is required.
  (The final `utils.py` casts each value to the type of the existing config value.)
- Never use `bool(some_string)` for coercion: every non-empty string is True.

## Optimizer internals

- An `Optimizer` holds three things: `defaults` (a template, read only when a group is created),
  `param_groups` (a list of dicts holding the live hyperparameters), and `state` (keyed by the Parameter *object*).
- Passing a flat parameter iterable wraps everything into ONE group, so `len(param_groups) == 1`
  is expected. That counts hyperparameter configurations, not parameters.
- Schedule the LR by writing `group["lr"]` before `step()`, which re-reads it on every call.
  There is no `opt.lr`, and mutating `opt.defaults` does nothing.

## Evaluation

- `model.eval()` only flips `self.training` recursively. Nothing in this package reads it, so for now
  it is a no-op. Keep it anyway: it becomes a silent bug the moment dropout is added.
- `torch.no_grad()` does the real work: measured **3.02 GB → 1.02 GB** for one forward.
- Restore `model.train()` in a `finally`.
- Seed the eval batches with a private `np.random.RandomState(seed)` built inside `evaluate`. The global
  seed can't do this. A global stream keeps advancing, so each eval would sample different text, and
  eval draws would consume training's randomness (changing `eval_interval` would change the training batches).

## Measured performance (Apple silicon, 24 GB, fp32)

| config | params | peak MPS | tok/s |
|---|---|---|---|
| 4L/16H, batch 32 | 22.7M | 4.09 GB | ~18,000 |
| 4L/16H, batch 64 | 22.7M | 9.54 GB | ~18,300 |
| 12L/8H, batch 32 | 47.6M | 10.22 GB | ~8,700 |

- Checkpoint size = params × 4 B × 3 (weights + Adam `m` + `v`) = **272 MB**. Don't checkpoint often.
- Batch size barely changes throughput (17.6k–18.3k tok/s from batch 16 to 64): the GPU is not saturated.

## First run + ablations (243k-token corpus)

- Loss starts at 9.247 ≈ ln(10000) = 9.21, which confirms a calibrated init.
- **Val bottoms at 3.256 (step 249, 8.4 epochs), then rises monotonically.** By step 599, train is
  0.443 and val is 4.248, a 3.81-nat gap. Classic memorization.
- Seven ablations (lr 3e-4/1e-3/3e-3, 2/4/8 layers, batch 16/32/64) all land within **0.077 nats** of
  each other (3.246–3.323). **The data sets the floor, not the architecture or the hyperparameters.**
  What varies is *when* the floor is hit (step 174–374) and how hard the run overfits afterwards.
  Aggressive configs (lr 3e-3, batch 64) arrive early and degrade worst.
- 8 layers beat 2 layers by 0.010 nats at 37% of the throughput. Not worth it at this scale.

## Dataset scale: the key finding

- The handout's MPS reference is exactly my `base.yaml`: `32 × 5000 × 256 = 40,960,000` tokens,
  36 min on mps, **val loss 1.80** (low-resource target ≤ 2.00). I was at 3.26 purely because I
  replayed 243k tokens 168 times, while the reference sees 41M distinct tokens (0.075 epochs).
- Compression: **4.105 chars/token**. The full train file, 2.07 GiB, becomes **542.7M tokens → 1.01 GiB**
  as uint16 (half the source size).
- **Training RAM does not scale with dataset size.** `np.memmap` pages in on demand. One batch
  touches 32 windows × 257 tokens = 16 KiB, i.e. ≤ 32 page faults per step regardless of file size.
  GPU memory depends only on batch/context/d_model/layers.
- **Encoding is where memory blows up.** A Python list of ints costs ~36 B/token (8 B pointer +
  28 B int object; CPython caches only ints ≤ 256). 543M tokens = **18.2 GiB** for the list, plus
  2.07 GiB for the source string, which is over 24 GB. Encode in chunks (`Tokenizer.encode_file_chunk` /
  `encode_iterable`), converting each chunk to uint16 before discarding it.
- Encode throughput measured at 1.55 MB/s single-process, so ~24 min for the full file.

## Next steps (as written then), with what happened

1. Chunk-encode the full TinyStories train + valid to `.bin`. **Done** (541.2M train tokens).
2. Re-run base at 5000 steps, target val ≤ 2.00. **Done: val 1.732.**
3. Overrides: add `list`/`None` branches and the unknown-key guard. Currently an override on
   `optim.betas` or `special_tokens` is silently ignored, which would invalidate an ablation. *Still open.*
4. Log the grad norm (have `gradient_clipping` return it instead of `None`) to see whether clipping
   actually fires at `max_l2_norm: 1.0`. *Still open.*
5. Track `best_val` and write `ckpt_best.pt`. With `checkpoint_interval: 5000`, the single saved
   checkpoint is the *most* overfit one. *Still open* (on the full corpus the last step was also the best).
6. Use `run_name` (a timestamp) so runs don't append into each other's `metrics.jsonl`. **Done.**
7. Experiments: `learning_rate` (incl. an attempt to diverge), `batch_size` (1 … 128), and the four
   architecture ablations (no RMSNorm, post-norm, NoPE, SiLU vs SwiGLU) behind config switches. **Done**,
   see [EXPERIMENTS.md](EXPERIMENTS.md). Decoding/generation: written (`generate.py`), sample in the README.
8. Tip from the handout: `torch.compile(model, backend="aot_eager")` on mps (Inductor is unsupported). *Not tried.*

## Smaller notes from the notebooks

- **BPE incremental update.** I first tried to patch the pair counts by hand, checking the neighbours
  left and right of each merge. That needed many special cases (the regression test was "shared-byte
  but non-touching pair must NOT be decremented"). The version I kept takes, for each changed word, the
  pairs after the merge minus the pairs before it, and adds that delta. I tested it against a
  brute-force recount on toy cases: `(a, a)` merges over `"lll"`, adjacent repeated merges, and two
  words collapsing into the same new word.
- **Tie-breaking.** An early bug: `max()` over the tied pairs compared token *ids*. The fix compares
  the pairs' *bytes*: `max(..., key=lambda x: (vocab[x[0]], vocab[x[1]]))`.
- **File modes.** In `"rb"` mode, `read` returns bytes and `seek`/`tell` count bytes. In text mode
  `tell` returns opaque cookies, not character positions. So chunk boundaries are found in binary
  mode and each chunk is decoded afterwards.
- **`.bin` vs `.npy` vs `.pkl`.** `.bin` (`tofile`) is raw values with no header: read it back with the
  right dtype, or you get silent garbage. `.npy` has a header (dtype, shape). `.pkl` stores arbitrary
  Python objects (vocab dict, merges list) and runs code on load, which is why `torch.load` now
  defaults to `weights_only=True`.
- **OpenWebText.** BPE training on OWT ran out of memory on this machine.
