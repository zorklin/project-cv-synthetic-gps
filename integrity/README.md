# Integrity checks

`SHA256SUMS` contains checksums for the immutable MCAP, metadata, and
calibration inputs. Run the verification from the repository root:

```bash
sha256sum --check integrity/SHA256SUMS
```

This file does not cover generated WSL artifacts. Each final or experimental
runner stores hashes for its own inputs and outputs in its run manifest.
