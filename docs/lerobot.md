# LeRobot (`libs/lerobot`)

Upstream: LeRobot by Hugging Face (Apache License 2.0, `libs/lerobot/LICENSE`). DiscoDemo uses it for datasets and
pi0.5 fine-tuning (`sft/`).

## Changed upstream files

- `src/lerobot/policies/pi05/modeling_pi05.py` (`resize_with_pad_torch`): float images are expected in [0, 1] and are
  padded with 0 before the model maps them to [-1, 1], so the padding band is black (-1) as in openpi and the
  pretrained checkpoint. Upstream padded with -1 before the mapping, which became -3.
- `src/lerobot/datasets/video_utils.py` (`VideoDecoderCache`): the decoder cache is an LRU of at most 100 decoders
  that closes the file handle of an evicted decoder. Upstream keeps every decoder open for the life of a
  DataLoader worker, which exhausts host memory on datasets with thousands of video files.
- 13 documentation files that are symlinks into `docs/` upstream (not shipped here) are regular copies with the same
  content: `policies/{act,diffusion,groot,smolvla,tdmpc,vqbet,wall_x}/README.md` and
  `robots/{earthrover_mini_plus,hope_jr,koch_follower,lekiwi,so100_follower,so101_follower}/*.mdx`.

Removed upstream content: `docs/`, `examples/`, `benchmarks/`, `tests/`, `media/`, CI, Docker and development files.
