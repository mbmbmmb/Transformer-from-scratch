# Cleanup log

What changed between my working copy and this repo, so it is clear which code is still exactly
what I wrote. The algorithms, numerics and module structure were left alone. The changes were:
renames for consistent naming, removal of dead code and clutter, `ruff format`, short
docstrings/comments, and four small bug fixes (listed below).

## How behaviour was checked

1. **Unit tests** (the course test suite in `tests/`, wired up through `tests/adapters.py`), run before
   and after the cleanup with the same environment:
   - before: **46 passed, 2 skipped**
   - after: **46 passed, 2 skipped**. The per-test PASSED/SKIPPED list is identical.
     The 2 skips are the memory-limit tokenizer tests, which only run on Linux (`rlimit`).
2. **Old-vs-new equivalence script.** With `torch.manual_seed(0)`, it builds the LM in five variants
   (baseline, `no_rmsnorm`, `post_norm`, `silu`, `no_rope`) and records the `state_dict` keys, every
   parameter's sum and a forward-pass output. It then trains BPE (serial and parallel, vocab 400) on
   `tests/fixtures/tinystories_sample.txt`. The md5 of all results is **identical** for the original
   and the cleaned code. So the random-init order, the parameter names (old checkpoints still load)
   and the BPE merges are unchanged.

## Renames

| before | after | why |
|---|---|---|
| `AdamW.py` | `adamw.py` | snake_case modules |
| `GenerateText.py` | `generate.py` | snake_case |
| `training_together.py` | `train.py` | shorter, clearer entry point (`scripts/*.sh` updated) |
| `transformer.py` / class `Transformer` | `transformer_block.py` / `TransformerBlock` | it is one block, not the model |
| class `Transformer_LM` | `TransformerLM` | PEP 8 class name |
| classes `multihead_self_attention`, `multihead_self_attention_with_rope` | `MultiHeadSelfAttention`, `MultiHeadSelfAttentionWithRoPE` | PEP 8 class names |
| `scale_dot_product_attention.py` / function | `scaled_dot_product_attention.py` / `scaled_dot_product_attention` | typo |
| `train_bpe_prallel()` | `train_bpe_parallel()` | typo |
| `pretokenization_example.py` | `pretokenization.py` | course-provided helper, now imported for real |
| local `scale_wt_casual` / `att_wt_casual` | `..._causal` | typo |

Attribute names (`transform_layers`, `att`, `ffn`, `rmsnorm1`, `linear`, `W`, `g`, `weights`, ...) were
**not** renamed, because they are the `state_dict` keys of the saved checkpoint.

## Files removed

| file | why |
|---|---|
| `cs336_basics/train_bpe_parallel.py` | an exact copy of the parallel pre-tokenization functions already in `train_bpe.py`, with a broken import (`from pretokenization_example import ...`) |
| `cs336_basics/SGD.py` | the handout's toy SGD example; not used by training (it also had a mis-placed f-string, `"fInvalid learing rate: {lr}"`) |
| `cs336_basics/ref/` (a screenshot) | reference image, not code |
| `configs/test.yaml` | an outdated copy of `base.yaml` without `paths`/`tokenizer`; `train.py` would crash on it |
| `SiLU` class in `swiglu.py` | unused: the SiLU ablation uses `SwiGLU(silu=True)` |
| course `README.md`, handout PDF, `CLAUDE.md`/`AGENTS.md`, `make_submission.sh`, `CHANGELOG.md` | course material, not my work |
| `data/` (14 GB), `runs/` (262 MB incl. a 260 MB checkpoint) | too large; the results are summarized in `docs/` |

## Clutter removed (per file)

- `adamw.py`: dead `alpha_t = 1` initialisation; formatting.
- `checkpointing.py`: `return None`, unused `Any` import.
- `cross_entropy.py`: docstring only.
- `data_loading.py`: the long tutorial comment block (toy examples of fancy indexing and memmap) was
  condensed into a docstring.
- `embedding.py`, `linear.py`, `rmsnorm.py`, `rope.py`, `softmax.py`, `scaled_dot_product_attention.py`:
  unused imports (`numpy` in linear), scratch comments (`# torch.block_diag(...)` in rope,
  `# output = self.W@x`), a wrong shape comment in linear, and a commented-out `d_v` line. In rmsnorm, the
  local variable `RMSNorm`, which shadowed the class name, became `normed`.
- `multihead_self_attention.py`: commented-out lines (`# self.device`, `# self.theta`, an old mask
  line), the fragment `# make sure t`, and the in-function `from .rope import RoPE` (moved to the top).
- `transformer_block.py`, `transformer_lm.py`: unused jaxtyping imports; in-function imports moved to the
  top (same construction order, verified by the equivalence script).
- `swiglu.py`: unused `Bool`, `Int` imports, the unused `SiLU` class.
- `train_bpe.py`: commented-out code (old `encode`, debug `print`s, a commented `train_bpe_prallel` body,
  stray `assert` scratch lines), unused `yaml` import, a duplicate `from pathlib import Path`, and a
  mid-file `import pickle` (moved to the top). Both `train_bpe` and `train_bpe_parallel` keep their
  own merge loop, as written.
- `train.py`: unused `yaml`, `torch.nn` imports; fixed the typo "backoward" in a comment.
- `generate.py`: duplicate `from cs336_basics.utils import load_config`.
- `utils.py`: an unreachable branch (`if new_val_str is None`: `str.partition` never returns `None`)
  and a commented-out line.
- `tests/adapters.py`: `sys.path.append('../cs336_basics')` hack, an unused top-level `Tokenizer` import,
  leftover `# raise NotImplementedError` lines and starter-code example comments. The imports were
  updated for the renames.
- `tests/test_model.py`: a debug `print` I had added to `test_linear` was reverted to the course version.
- `configs/base.yaml`: removed keys that no code reads (`training.dtype`, `training.compile`,
  `logging.checkpoint`, `logging.log_interval`, `logging.wandb`, `logging.wandb_project`, `resume_from`)
  and commented-out blocks. Kept every key that the code or the sweep scripts use.
- `pyproject.toml`: removed the unused dependencies `wandb`, `einx` and `tqdm`; `uv.lock` was regenerated.

## Bug fixes

| where | what was wrong | fix | evidence |
|---|---|---|---|
| `cs336_basics/generate.py:55-56` | `--output_length` and `--temperature` had no `type=`, so values given on the command line arrived as strings. `ids.shape[-1] - start_len < "50"` raises `TypeError` (only the defaults worked). | `type=int`, `type=float` | reproduced the `TypeError` with the original argparse setup; tests unchanged |
| `cs336_basics/generate.py:53` | the default `--config runs/base/config.yaml` is a hand-written record of the run, with paths that no longer exist | default `configs/base.yaml`, whose `logging.out_dir` (`runs/base`) and tokenizer paths match where the trained checkpoint and tokenizer live | — |
| `cs336_basics/train_bpe.py:243` (`split_special`) | only `None` was treated as "no special tokens". With `special_tokens=[]` the pattern is `""`, so `re.split("()", text)` splits between every character, and `encode` fails with `KeyError: b''` | `if not special_tokens: return [lines]` | before: `Tokenizer(vocab, [], special_tokens=[]).encode("hi")` → `KeyError: b''`; after: `[104, 105]`; tests unchanged |
| `pyproject.toml` | `utils.py` imports `yaml`, but PyYAML was not a declared dependency (it only arrived through `wandb`) | added `pyyaml>=6` | `uv lock` resolves it |

## Not changed: suggestions to work on

These are inefficiencies and probable bugs I noticed while cleaning. They are left as they were so the
code stays mine. Line numbers refer to the cleaned files.

### Efficiency

1. **BPE merge loop is O(#words) per merge.** `train_bpe.py:82` (`get_keys_to_replace`) scans every
   distinct pre-token for every merge, and `train_bpe.py:53` (`get_top_pair_lex`) scans every pair
   twice per merge (once for `max`, once for the ties). Ideas: keep an index `pair → set of words
   containing it`, so a merge only touches affected words; keep pair counts in a max-heap with lazy
   deletion (or bucket by count) instead of a full scan.
2. **Tokenizer.encode merges by rebuilding tuples.** `train_bpe.py:263-300`: each merge step rescans
   the whole pre-token (`get_merge_pair`) and rebuilds the tuple by slicing (`merge_single_idx`), which
   is quadratic in pre-token length. The cache hides this for common words. The cache itself
   (`train_bpe.py:315`) grows without bound over a 2 GB corpus; consider an LRU cap.
3. **`split_special` rebuilds and re-sorts the special-token pattern on every `encode` call**
   (`train_bpe.py:245-246`). `encode_iterable` calls `encode` once per line, so this happens per line.
   It could be built once in `Tokenizer.__init__`.
4. **Cross-entropy computes the row max twice** (`cross_entropy.py:15-16`). `M_flat` equals
   `M.squeeze(-1)`.
5. **Gradient clipping forces a device → host sync** (`gradient_clipping.py:20`). `math.sqrt` on a
   tensor calls `.item()` every step, and the norm is accumulated parameter by parameter. Staying in
   torch (e.g. stacking the per-parameter norms) avoids the per-step sync.
6. **Generation builds an autograd graph and recomputes everything** (`generate.py:43-44`): no
   `torch.no_grad()`/`model.eval()`, and each new token re-runs the full sequence (no KV cache).
7. **Causal mask re-allocated every forward** (`multihead_self_attention.py:64, 142`). It could be a
   registered buffer sized `max_seq_len` and sliced, like the RoPE tables.
8. **AdamW loops over parameters in Python** (`adamw.py:26`). This is fine for a from-scratch
   version; `torch._foreach_*` ops would batch it.
9. **Unused RMSNorm parameters in the ablations.** With `no_rmsnorm=True`, `TransformerBlock` still
   creates `rmsnorm1`/`rmsnorm2` (`transformer_block.py:64`). They receive no gradient, so it is harmless
   apart from wasted parameters and optimizer state.

### Probable bugs / sharp edges

1. **SwiGLU init std looks off:** `std_ffn = 2 / sqrt(d_ff + d_model)` (`swiglu.py:34`). The usual
   truncated-normal init is `sqrt(2 / (d_in + d_out))`, as `linear.py` does. For d_model 512 and
   d_ff 1344 that is 0.046 vs 0.033, about 1.4× too large. The tests can't catch it, because they pass
   the weights in (the "fixtures never exercise the init path" lesson in LEARNING_NOTES).
2. **The two attention classes use different init truncation:** `MultiHeadSelfAttention` clips at
   ±1σ (`multihead_self_attention.py:31`), while `MultiHeadSelfAttentionWithRoPE` clips at ±3σ.
   Only the RoPE class is used in training.
3. **No-RMSNorm runs explode at init** (loss ~7e5 at step 49, see EXPERIMENTS.md). This may be the
   expected result of the ablation, but it is worth checking how much of it comes from the init scales
   above rather than from the missing norms.
4. **`generate.py` doesn't crop the context.** Once `prompt + output_length` exceeds
   `context_length` (256), RoPE indexes past its `cos`/`sin` table and fails. Feed only the last
   `context_length` tokens.
5. **`load_checkpoint` uses `torch.load(src)` without `map_location`** (`checkpointing.py:28`), so a
   checkpoint saved on `mps` will not load on a CUDA/CPU-only machine.
6. **`generate.py` looks for `<out_dir>/ckpt_{max_iters-1}.pt`, but `train.py` writes to
   `<out_dir>/<run_name>/`.** For a new run, pass a config whose `logging.out_dir` points at the run folder.
7. **Config overrides silently ignore lists and `None`** (`utils.py:6`): e.g. `optim.betas=...` or
   `logging.run_name=foo` (the default is `null`) does nothing. Unknown keys raise a bare `KeyError`.
   (Already on my next-steps list.)
8. **`Tokenizer.__init__` mutates the caller's `vocab` dict** when it adds missing special tokens
   (`train_bpe.py:309`), because `self.vocab = vocab` is not a copy.
9. **`read_pkl` creates the parent directory** (`train_bpe.py:235`) before reading. That is harmless,
   but it hides a wrong path until the open fails.
10. **The `train_bpe` / `train_bpe_parallel` merge loops are duplicated.** A shared helper would keep
    them from drifting apart.
11. **`train_bpe` with `special_tokens=[]`** has the same empty-pattern problem as `split_special` had
    (`train_bpe.py:122` in `get_chunk_word_count`, `train_bpe.py:175` in `train_bpe`).
12. **2000-step sweeps kept `cosine_cycle_iters: 5000`**, so the LR never decays (see EXPERIMENTS.md).
    For a future sweep, set `lr_schedule.cosine_cycle_iters` together with `training.max_iters`.
