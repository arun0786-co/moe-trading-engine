# moe-trading-engine

A mixture of experts model that predicts whether the Nifty 50 goes up tomorrow,
with a walk forward evaluation against two simple baselines.

I wrote most of the engine in June 2026 and never got it running end to end.
In November 2026 I fixed it, wrote the evaluation, and ran it. The fixes are
separate commits after the June code, so you can see what changed.

**Result: it does not beat buy and hold.** Details below.

## model

- three experts, each reads the last 30 days of price, volume, trend, momentum
  and volatility features:
  - liquid time constant neural ODE (`engine/models/expert_ode.py`, torchdiffeq)
  - hyper LSTM, a small LSTM that rescales the main LSTM's weights every step
    (`expert_hyper_lstm.py`)
  - selective state space model, mamba style, with a parallel scan (`expert_ssm.py`)
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
- 3 seeds for the MoE. All runs on one RTX 2080 Ti (`jobs/eval_cuda.sh`)

Up days are 54.3% of the test period, so "always up" scores 0.543 accuracy.

### results, 2016 to 2026

Engine costs. MoE is the mean of 3 seeds, range in brackets.

| | accuracy | total return | CAGR | Sharpe | max drawdown | days in market |
|---|---|---|---|---|---|---|
| buy and hold | 0.543 | +240% | 13.8% | 0.89 | -37.7% | 100% |
| logistic regression | 0.536 | +126% | 9.0% | 0.68 | -33.4% | 78% |
| MoE | 0.520 (0.511 to 0.526) | +85% (+48 to +120%) | 6.6% | 0.52 (0.36 to 0.64) | -33.5% | 67% |

At 0.1% per side:

| | total return | CAGR | Sharpe |
|---|---|---|---|
| buy and hold | +239% | 13.8% | 0.89 |
| logistic regression | +76% | 6.2% | 0.49 |
| MoE | +34% (+8 to +66%) | 3.0% | 0.28 |

What this says:

- accuracy is below "always up" in all 3 seeds. The model has not learned
  anything useful about tomorrow's direction
- it is also worse than logistic regression on the same features
- the seeds disagree a lot (+48% to +120%), so most of the spread is noise
- trading in and out about 300 times in under 10 years costs a lot at real
  Indian delivery costs

Per fold numbers for each seed are in `results/eval_nifty_s*.json`.

## bugs fixed before the evaluation

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
python scripts/evaluate.py --device cuda --seed 0 --out results/eval_nifty_s0.json
```

One seed takes about 8 hours on a 2080 Ti, mostly the later folds where the
training window is longest. `--folds 1 --epochs 1` is a quick check.
`--reuse N` loads saved checkpoints for the first N folds instead of training.

Not financial advice.

## license

MIT
