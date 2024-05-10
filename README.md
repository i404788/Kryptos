# Kryptos

This is a single python3 file which is able to crack K1-K3 (and reproduce key extractions) and additionally attempts the same techniques (also chained) for K4.

Current status of K4:

- [x] Vig (or key extraction); up to key length 38 is not possible
- [x] (skip/rot) Transpose incl reverse -> Vig/Cesear; is not possible decoder for key lengths <30
- [x] Vig -> Tranpose; also not possible for key lengths <32

Vig with higher key length is possible but impossible to validate is it doesn't allow for extrapolation from the known values.
Vig key extraction did not give a extrapolatable key.

> NOTE: not possible here means that it has been exaustively cracked: vig using the standard table for certain key length; transpose using all possible skips (2->N)

If you wish to contribute feel free to post your results, code or ideas. I've added most of the hints inside the file (some already used for validation) for easy copying.

The primary structure used is numpy's u8 typed arrays which effectively give ascii encoded values, then using `set_printoptions` we ensure values in the right range are printed as characters.

This allows for quick iteration and debugging. We recommend checking the K2/K3 cracking code (and helper functions) to get an understanding of how to use it.

It has some quirks like needing to subtract, mod, then add `a`  in order to do modular arithmatic.

Excedingly useful functions: 
- `apply_table_idxs`
- `find_repeats`
- `coded_cesar` (for cracking vig)
- `histogram` (for statistical analysis)
- `pad_a_to_b` (for stream-cipher key extension)


