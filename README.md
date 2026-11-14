# make.fun entry: polymarket-faa-attribution-01

A Python 3 program (standard library only) that forecasts the Yes price of every market of Polymarket's FAA outage attribution family, minute by minute, 4 hours ahead: for each case file in the input directory it writes five quantiles (0.05, 0.25, 0.5, 0.75, 0.95) for each of the 240 minutes after the checkpoint.

Method: per market, the empirical distribution of that market's own recent price changes at several horizons, interpolated across horizons, added to the last price, with a small floor spread so quiet markets never forecast zero width.

Run locally:

    mkdir -p in out && cp case.in in/ && MAKEFUN_INPUT_DIR=in MAKEFUN_OUTPUT_DIR=out python3 main.py
