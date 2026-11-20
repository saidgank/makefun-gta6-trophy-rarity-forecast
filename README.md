# makefun-gta6-trophy-rarity-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/saidgank/makefun-gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/saidgank/makefun-gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A forecaster for the earn rate of every Grand Theft Auto VI (PS5) trophy on Exophase at 2026-11-26T00:00:00Z, one week after launch. `main.py` (Python 3.13, standard library only) reads a case — every read the house recorded of the game's trophy page before the checkpoint — and writes five quantiles per trophy, scored by pinball loss on the logit scale.

The model works in logit space and in earners space (rate × players tracked), never in raw percent alone. It compresses the history to distinct page states (the page only changes when Exophase refreshes it), fits the tracked pool's growth as a damped power law, and per trophy blends a recency-weighted logit trend in log-time with an earners model whose growth elasticity is estimated against the pool — the two forces on every rate are progression and dilution. A story-funnel flow lift keeps late chapters and story missions from sitting at zero before the player wave reaches them, and the spread combines a random-walk term, the fit's own prediction error and the two models' disagreement, widened for trophies pinned at the printed floor.

Calibration on the game's own recorded launch: backtesting from the launch-night page to the real page 37 hours later, the forecaster's mean pinball loss is 0.76 against the house baseline's 2.93 (split-half across trophies: 0.78–0.89 against 3.03–3.44); on a 6.6-hour holdout in the calm regime it scores 0.077 against the baseline's 0.063, the honest cost of a real spread at a horizon where almost nothing moves.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=kvcj2dy8&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
