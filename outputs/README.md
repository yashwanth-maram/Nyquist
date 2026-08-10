# Restored test outputs

400 restored images from `Test_NoisyLR/`, produced by `evaluate.py` with the
submitted weights and 8x self-ensemble.

Each file is 256x256 float32 in [0,1], named identically to its input
(`000000.npy` .. `000399.npy`).

To regenerate:

    python evaluate.py --input-dir /path/to/Test_NoisyLR --output-dir outputs/
