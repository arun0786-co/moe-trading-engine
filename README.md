# moe-trading-engine

A mixture of experts model that predicts whether the Nifty 50 goes up tomorrow,
with a walk forward evaluation against two simple baselines.

I wrote most of the engine in June 2026 and never got it running end to end.
In November 2026 I fixed it, wrote the evaluation, and ran it. The fixes are
separate commits after the June code, so you can see what changed.

**Result: it does not beat buy and hold, and it shows no sign of predicting
the next day at all.** Details below.

## model

- three experts, each reads the last 30 days of price, volume, trend, momentum
  and volatility features:
  - liquid time constant neural ODE (`engine/models/expert_ode.py`, torchdiffeq)
  - hyper LSTM, a small LSTM that rescales the main LSTM's weights every step
    (`expert_hyper_lstm.py`)
  - selective state space model, mamba style, with input dependent step sizes
    (`expert_ssm.py`)
- a router that looks at regime features (rolling volatility, ADX, Hurst
  exponent, autocorrelation, volume regime) and picks the expert. Soft mix with
  Gumbel softmax while training. At test time it uses one expert when the router
  is confident and a soft mix otherwise (`router.py`)
- two heads: probability of an up day, and the size of the move (`projection.py`)
- about 262k parameters

## evaluation

`scripts/evaluate.py`

- data: Nifty 50 daily bars from yfinance (`^NSEI`), 2010-01-04 to 2026-11-20
- 11 walk forward folds with an expanding window. Fold 0 trains on 2010 to 2014,
  picks the epoch on 2015 and tests on 2016. The last fold tests on 2026 up to
  20 Nov. 2352 test days in total
- a fresh model per fold, early stopping on validation accuracy
- strategy: hold the index tomorrow if p(up) > 0.5, else stay in cash
- costs on each buy and sell, two settings:
  - the engine's cost model (`engine/backtesting/costs.py`): 0.043% to buy,
    0.053% to sell
  - a pessimistic 0.1% per side (delivery STT alone is 0.1%)
- baselines: buy and hold, and logistic regression on the last day's features
- 3 seeds for the MoE

Up days are 54.3% of the test period, so "always up" scores 0.543 accuracy.

### results, 2016 to 2026

Current code (after the expert fixes listed below), engine costs. MoE is the
mean of 3 seeds, range in brackets. Trained on an M4 Pro CPU, about 3.3 hours
for all 3 seeds in parallel.

| | accuracy | total return | CAGR | Sharpe | max drawdown | days in market |
|---|---|---|---|---|---|---|
| buy and hold | 0.543 | +240% | 13.8% | 0.89 | -37.7% | 100% |
| logistic regression | 0.536 | +126% | 9.0% | 0.68 | -33.4% | 78% |
| MoE | 0.509 (0.503 to 0.516) | +36% (+29 to +45%) | 3.3% | 0.31 (0.26 to 0.35) | -37.6% | 64% |

At 0.1% per side:

| | total return | CAGR | Sharpe |
|---|---|---|---|
| buy and hold | +239% | 13.8% | 0.89 |
| logistic regression | +77% | 6.2% | 0.49 |
| MoE | -4% (-11 to +5%) | -0.4% | 0.04 |

First run, before the expert fixes (RTX 2080 Ti, engine costs):

| | accuracy | total return | Sharpe |
|---|---|---|---|
| MoE | 0.520 (0.511 to 0.526) | +85% (+48 to +120%) | 0.52 (0.36 to 0.64) |

What this says:

- accuracy is below "always up" in every seed of both runs
- with the fixed experts no seed is better than a coin flip (binomial test
  against 0.5: p = 0.45, 0.76, 0.12 over 2352 days). The first run was a bit
  higher, but its experts were broken, so I read that as noise too
- it is worse than logistic regression on the same features
- trading in and out about 300 times in under 10 years costs a lot at real
  Indian delivery costs

Per fold numbers are in `results/eval_v2_s*.json` (current code) and
`results/eval_nifty_s*.json` (first run).

### looking inside the model

`scripts/diagnose.py` loads every saved fold model and checks it on the test
year (`results/diagnose_v2.json`). Accuracy over all 2352 test days, per seed:

| | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| routing used at test time | 0.508 | 0.503 | 0.516 |
| soft mix of all experts | 0.510 | 0.510 | 0.521 |
| liquid ODE alone | 0.530 | 0.508 | 0.503 |
| hyper LSTM alone | 0.489 | 0.504 | 0.520 |
| selective SSM alone | 0.518 | 0.520 | 0.521 |

- the router trains on a soft mix but at test time sends a day to one expert
  when it is more than 70% sure (25% of days). Using the soft mix instead
  helps by about 0.005, which is within the noise
- no expert on its own is reliably better than another
- early stopping picks epoch 0 to 2 in 7 of the 33 fold models. With about
  250 validation days, validation accuracy is mostly noise, so the "best"
  epoch is close to random

So the architecture is not the problem. There is no next day signal in these
features that this setup can find.

## bugs fixed

The June code had never run end to end. Things that were wrong:

- the training script imported classes that did not exist
- mixed precision used the old `torch.cuda.amp` api and crashed. AMP is now off;
  the model is small and the ODE solver is sensitive to precision
- the model returned a named tuple but the trainer expected a dict
- predictions were `[B, 1]` and targets `[B]`, so the loss and accuracy
  broadcast to `[B, B]` and compared every prediction with every target
- the target was off by one: it predicted the day after tomorrow.
  `tests/test_dataset.py` fails on the old code
- `.gitignore` had `models/`, which also hid `engine/models/`
- the ODE expert used `odeint_adjoint`. Its backward pass re-runs the
  dynamics after the time loop is over, when the stored input is the last
  day's, so 16 to 85% of the gradient was wrong. It now backprops through the
  solver (`tests/test_ode.py` checks against finite differences)
- the SSM's "parallel scan" was never finished. The leftover up sweep wrote
  into the decay terms in place before the sequential scan ran, so the state
  was partly wiped every other day (`tests/test_ssm.py`)
- the downloader crashed when the cached data ended on a Friday and the run
  asked for data up to a weekend. It now uses the cache
- my first version of the evaluation started every fold flat, so buy and
  hold paid for 11 buys instead of 1 (+238% instead of +240%). The folds are
  now joined and scored as one run, with a test

## not tested

These modules are from June and nothing here runs them: the paper trader
(`engine/execution`), risk rules (`engine/risk`), news sentiment
(`engine/data/sentiment.py`), the bar by bar backtester
(`engine/backtesting/backtester.py`), the online learner and the trade logger.
I removed the backtest and daily pipeline scripts because they did not match
the engine's api.

## run it

```
pip install torch numpy pandas scipy yfinance ta pyyaml torchdiffeq pytest
python -m pytest -q tests
python scripts/evaluate.py --device cpu --threads 3 --seed 0 --out results/eval_v2_s0.json
python scripts/diagnose.py --device cpu --ckpt-dir models/eval_s{seed}
```

After the speed fixes (the hyper LSTM no longer builds a weight matrix per
sample per step, the SSM scan no longer allocates a full tensor per step in
backward) one seed takes about 3 hours on 3 CPU threads. `--folds 1 --epochs 1` is a quick check.
`--reuse N` loads saved checkpoints for the first N folds instead of training.

Not financial advice.

## license

MIT
