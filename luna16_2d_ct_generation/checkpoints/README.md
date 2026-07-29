# Model checkpoint metadata

The trained checkpoints are intentionally excluded from Git because both files
exceed GitHub's 100 MB single-file limit. Do not bypass this limit with a normal
`git add`; publish them as GitHub Release assets or configure Git LFS first.

| Model | Expected path | Size (bytes) | SHA-256 |
|---|---|---:|---|
| 64×64 baseline, 3000 steps | `checkpoints/latest.pt` | 120,564,171 | `eacec0c5c56391cb1e18f84e5ae8000e103528ed4b5f89ae8bed5043595c1bd3` |
| 128×128 improved, 10000 steps | `checkpoints/improved_128/latest.pt` | 120,627,275 | `2318aa7bb8a2a72e1526b4225829c1ac847541b38eedcd23f898b1d760309ac3` |

Reproduce the files with the training commands in the project README, then
verify them on Linux with:

```bash
sha256sum checkpoints/latest.pt checkpoints/improved_128/latest.pt
```

If Git LFS is configured by the repository owner:

```bash
git lfs track "luna16_2d_ct_generation/checkpoints/**/*.pt"
```

Alternatively, attach only the two final `latest.pt` files to a tagged GitHub
Release. Intermediate and smoke-test checkpoints should not be published.
