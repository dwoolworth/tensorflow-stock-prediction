Rudimentary prediction of stock value over time
---

This script runs in about 15 seconds on my workstation.  If you do not have CUDA cores, it may take
up to 1 minute to finish building the model.

This is just an example, the code was stolen, but I've tweaked it just a bit and get a little better
results.  If additional datum were added the results would surely become more accurate.

```
# To install dependencies
$ python -m pip install -r requirements

# To run specify the ticker and whether the results should be plotted
$ python rubmytensor.py AAPL y

```

