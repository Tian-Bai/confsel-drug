# confsel-drug

Conformal selection vs eCounterScreening (Sheridan et al., 2015) for compound screening/counterscreening.

## Contents

- `sheridan-vs-conformal.py`: The experiment comparing conformal selection versus eCounterScreening
- `plot-all.py`: With all results available, generate the FDP and power plots

Example:
```bash
python3 sheridan-vs-conformal.py $dataset $sample_size $seed
python3 plot-sheridan-vs-conformal.py $sample_size $n_itr
```