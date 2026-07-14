# Hedge Fund Portfolio Risk & Returns Simulator

A Pandas/NumPy-driven tool for analyzing the risk and return profile of a
multi-stock portfolio, with a Scikit-learn trend layer and an interactive
Streamlit dashboard on top.

---

## Why I built this

I'm a 3rd-semester student, and I built this right after finishing NumPy and
partway into learning Pandas. I wanted a project that would actually force
me to understand the "why" behind the code, not just get comfortable with
the syntax. Quant finance turned out to be a good fit for that — it's one of
the few areas where a handful of Pandas/NumPy methods map almost directly
onto real, well-established math: `.pct_change()` is the foundation of
return-based risk analysis, `np.dot()` is how a portfolio's daily return
actually gets computed. Building this felt like seeing what those methods
were *for*, rather than just practicing them in isolation.

## What it does

You give it a portfolio (AAPL, MSFT, GOOGL, AMZN by default, adjustable
in code) and a set of weights, and it walks through five layers of
analysis, each one building on the last:

1. Pulls 5+ years of real historical price data via `yfinance`, cleans it,
   and converts prices into daily returns.
2. Combines the individual stocks into a single portfolio return series,
   using matrix multiplication.
3. Measures portfolio-level risk properly — accounting for how the stocks
   move *together*, not just their individual volatility.
4. Quantifies downside risk with Historical Value at Risk (VaR): a direct
   answer to "on the worst 5% or 1% of days, how much could this portfolio
   have lost?"
5. Projects where that risk is trending, using a simple regression model —
   deliberately scoped to risk, not price.

All of this sits inside a Streamlit dashboard where you can drag weight
sliders and watch every number and chart update live.

---

## A look at it

*(Add your own screenshots here before submitting — this section is
usually the first thing a reviewer actually looks at.)*

```
Some Screenshots of the dashboard, graph and heatmap are attached in screenshots file
```

---

## The math, explained simply

I didn't want any number on this dashboard that I couldn't explain if
someone asked me to walk through it. Here's the reasoning, kept light but
not hand-wavy.

### Returns, not prices

Raw prices aren't directly comparable across stocks — AAPL trades around
$200, GOOGL around $180, so "dollars moved" doesn't mean the same thing for
each. Percentage returns fix that:

$$r_t = \frac{P_t - P_{t-1}}{P_{t-1}}$$

Returns are also *stationary* — their statistical behavior (mean, spread)
stays roughly consistent over time, unlike prices, which trend. That's why
every formula below operates on returns, computed with Pandas'
`.pct_change()`.

### Combining stocks into a portfolio

If a portfolio holds several stocks with weights $w_1, w_2, ..., w_n$
(summing to 1), the portfolio's return on any given day is just the
weighted sum of each stock's return that day:

$$R_{portfolio} = w_1 r_1 + w_2 r_2 + ... + w_n r_n$$

Computed across thousands of days at once, this is exactly what
`np.dot(returns_matrix, weights)` does — one line of NumPy standing in for
what would otherwise be a loop over every day and every stock.

### Portfolio risk isn't just averaged individual risk

This is the part that surprised me most. A portfolio's risk isn't the
weighted average of each stock's own volatility — it depends on how the
stocks move *relative to each other*. Two stocks that always rise and fall
together don't diversify much; two that move somewhat independently cancel
out some of each other's swings. That relationship is captured by the
covariance matrix, and portfolio variance is computed as:

$$\sigma_p^2 = w^T \Sigma w$$

where $\Sigma$ is the covariance matrix of returns. When I actually ran
this on real data, it showed up as a concrete result: my equal-weighted
portfolio's volatility came out *lower* than any single stock in it — a
direct, numerical demonstration of diversification, not just something I
read about in a textbook.

### Historical VaR: reading the real distribution instead of assuming one

Value at Risk answers a specific, useful question: "on the worst X% of
days historically, how much did the portfolio lose?" There are a couple of
ways to estimate it, and the difference between them matters. A
*parametric* approach assumes returns follow a normal distribution and
estimates VaR from the mean and standard deviation. *Historical* VaR skips
that assumption entirely — it just sorts the real historical returns and
reads off the actual percentile:

```python
var = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
```

I compared both on my own data, and the parametric estimate at 99%
confidence came out noticeably less severe than the historical one. That
gap reflects something real: market returns tend to have "fatter tails"
than a normal distribution predicts — extreme days happen more often than
a bell curve would suggest — and Historical VaR captures that because it
never assumes a shape for the data in the first place.

### Projecting the trend, honestly

For the machine learning layer, I fit a Ridge regression on the portfolio's
trailing rolling volatility (a 30-day rolling standard deviation,
annualized) and extrapolate it 30 days forward. Ridge, specifically,
because its regularization term keeps the fitted line from overreacting to
a single noisy spike in recent volatility, which matters when the whole
point is reading a general trend rather than chasing short-term noise.

The more important decision, though, was *what* to project. This model
only ever looks at a rolling risk metric — never at price. A simple linear
model has no real basis for predicting where a stock will trade next month,
and building something that implied otherwise would have overstated what
this kind of model can actually do.

---

## Tech stack

| Layer | Tool | What it's doing |
|---|---|---|
| Data | `yfinance` | Historical price data, split/dividend-adjusted |
| Cleaning & returns | `pandas` | `.ffill()`, `.pct_change()`, `.rolling()` |
| Portfolio math | `numpy` | `np.dot`, covariance matrix, `np.percentile` |
| Trend projection | `scikit-learn` | `Ridge` regression |
| Interface | `streamlit` | Reactive dashboard, cached data loading |
| Visualization | `plotly` | Correlation heatmap |

## Project structure

```
hedge_fund_sim/
├── src.py                # Full analysis, run top to bottom from the terminal
├── dashboard.py           # Same underlying logic, wrapped in an interactive UI
├── requirements.txt
├── README.md
└── screenshots/
```

The core calculation functions are identical between `src.py` and
`dashboard.py` on purpose — the script is where I built and checked the
logic step by step, and the dashboard just wraps those same functions in a
reactive interface.

---

## Running it

```bash
pip install -r requirements.txt
```

**See the raw output in the terminal:**
```bash
python src.py
```

**Run the interactive version:**
```bash
streamlit run dashboard.py
```
Then adjust the sliders in the sidebar — every metric and chart updates
live as you do.

---

## What I learned building this

A few things stuck with me more than I expected:

- **Vectorization changes how you think about the problem, not just how
  fast the code runs.** Once `np.dot(returns_matrix, weights)` replaced
  what would have been a nested loop, the code started to look like the
  formula itself, which made the underlying math easier to hold in my head,
  not just faster to execute.
- **The most important decision here wasn't a line of code.** It was
  deciding to keep the ML layer scoped to a risk trend instead of letting
  it imply anything about future prices. That felt like the actual judgment
  call in this project, more than any individual function.
- **Real data behaves differently from tutorial data.** Handling missing
  values properly, keeping weights aligned to the right return columns, and
  catching a divide-by-zero edge case in the slider normalization — none of
  that comes up until you're working with something real.

## Known limitations

- Historical VaR assumes the future statistically resembles the past — it
  won't anticipate a genuinely unprecedented event.
- The volatility trend is a linear extrapolation over a fixed 252-day
  window. It's meant to indicate direction, not to forecast precisely, and
  I've tried to be clear about that throughout rather than overstate it.
- `END_DATE` is currently hardcoded rather than computed from the current
  date — a small fix worth making for longer-term use.
- The portfolio is fixed at four stocks with weights between 0–100% each;
  generalizing to an arbitrary number of holdings would be a natural next
  step.

---

If you're reviewing this as part of an application — thank you for reading
this far. I tried to make sure every decision here is something I can
actually explain, not just something that happened to run without errors.
