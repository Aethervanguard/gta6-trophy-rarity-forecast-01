[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Aethervanguard/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Aethervanguard/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

# gta6-trophy-rarity-forecast-01 — "Forecast how many GTA VI players earn each trophy in week one"

Entry for the make.fun bounty `gta6-trophy-rarity-forecast-01`: forecast, for
every PlayStation 5 trophy of Grand Theft Auto VI, its earn rate on Exophase
at 2026-11-26T00:00:00Z, as five quantiles (0.05, 0.25, 0.5, 0.75, 0.95) per
trophy, from the house's recording of the trophy page up to the deadline.

## The model

Launch week is when earn rates move fastest, so this is a trend forecaster,
not a persistence one. Per trophy it fits a line of the logit earn rate
against log time (`ln(1 + hours/12)` since release) over the last 24 hours of
reads, shrinks the slope toward its tier's median slope, caps the slope, and
anchors the forecast on the fitted level at the last read. The spread is a
logit normal built from the fit's residuals, the slope's standard error, and
a random-walk diffusion over the horizon. Trophies sitting at the rounding
floor (0.01% or below) get a lifted median and a wider band, because week
one is when mid-game trophies switch on. See `main.py` for the full
description.

## Run

```sh
python3 main.py
```

Reads every `*.in` in `$MAKEFUN_INPUT_DIR` (else `/input`) and writes one
`*.out` per case to `$MAKEFUN_OUTPUT_DIR` (else `/output`). Python 3.13,
standard library only, no network, deterministic.

## License

MIT
