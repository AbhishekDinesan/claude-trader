# Stock Analysis Tool

A self-learning stock analysis tool that uses technical analysis, momentum screening, and sentiment analysis to find high-potential opportunities. **It learns from its mistakes automatically!**

## Features

### Technical Analysis
- **RSI** - Identify oversold/overbought conditions
- **MACD** - Catch bullish/bearish crossovers
- **Bollinger Bands** - Spot squeezes and breakouts
- **Moving Averages** - Golden/death cross detection
- **Volume Analysis** - Identify volume spikes

### Autonomous Learning
- Tracks all predictions automatically
- Evaluates outcomes after 5-10 days
- Learns which indicators actually work
- Adjusts scoring weights based on performance
- Gets smarter over time!

### Sentiment Analysis
- Reddit sentiment (WSB, stocks, investing)
- News headline analysis
- Social buzz scoring

## Quick Start

### Install
```bash
cd trader
pip install -r requirements.txt
```

### Run
```bash
# Quick scan
python main.py scan --fast

# Scan AND track predictions for learning
python main.py scan --fast --track

# Analyze one stock
python main.py analyze NVDA

# Build a portfolio
python main.py portfolio
```

## Self-Learning System

### How It Works

```
1. Run scans with --track  -->  Predictions logged to database
2. Wait 5+ days            -->  Let trades play out
3. Run learning cycle      -->  Check what actually happened
4. Weights adjusted        -->  Better predictions next time!
```

### Learning Commands

```bash
# Log predictions during scan
python main.py scan --fast --track

# Run learning cycle (evaluate past predictions)
python main.py learn cycle

# Check learning statistics
python main.py learn stats

# View learned weights
python main.py learn weights
```

## Automated Deployment (GitHub Actions)

Deploy this to run automatically - completely FREE!

### Setup Steps

1. **Create a GitHub repository**
   ```bash
   cd trader
   git init
   git add .
   git commit -m "Initial commit"
   ```

2. **Push to GitHub**
   ```bash
   gh repo create stock-analyzer --public --source=. --push
   ```
   Or create repo on github.com and push manually.

3. **Enable GitHub Actions**
   - Go to your repo on GitHub
   - Click "Actions" tab
   - Enable workflows

4. **That's it!** The system will now:
   - Run daily scans at 9:30 AM ET (market open)
   - Log all predictions automatically
   - Run weekly learning cycles on Sundays
   - Commit results back to the repo

### Manual Trigger

You can also trigger runs manually:
1. Go to Actions tab
2. Select "Auto Stock Scanner & Learner"
3. Click "Run workflow"
4. Choose: `scan`, `learn`, or `full`

## Commands Reference

| Command | Description |
|---------|-------------|
| `scan --fast` | Quick market scan |
| `scan --fast --track` | Scan + log for learning |
| `analyze SYMBOL` | Deep dive on one stock |
| `portfolio` | Build a diversified portfolio |
| `momentum` | Find momentum leaders |
| `breakouts` | Find breakout candidates |
| `learn cycle` | Run learning evaluation |
| `learn stats` | Show learning statistics |
| `learn weights` | Show learned weights |

## Signal Guide

| Signal | Score | Meaning |
|--------|-------|---------|
| STRONG BUY | 75+ | Multiple bullish factors aligned |
| BUY | 60-74 | Favorable setup |
| HOLD | 40-59 | Wait for better entry |
| SELL | 25-39 | Consider exiting |
| STRONG SELL | <25 | Avoid |

## Project Structure

```
trader/
├── main.py               # CLI interface
├── scanner.py            # Unified scanner
├── technical_analysis.py # RSI, MACD, etc.
├── screener.py           # Stock screener
├── sentiment.py          # Reddit/news sentiment
├── learning.py           # Self-learning system
├── scheduler.py          # Automated scheduler
├── config.py             # Settings
├── .github/workflows/    # GitHub Actions
│   └── auto-learn.yml    # Automation config
├── results/              # Scan results (auto-generated)
├── learning_data.db      # Predictions database
└── learned_weights.json  # Learned model weights
```

## How Learning Improves Performance

After ~50-100 tracked predictions, the system:

1. **Identifies accurate indicators** - If RSI oversold correctly predicts bounces 70% of the time, it weights RSI higher

2. **Adjusts signal weights** - If BUY signals have 60% win rate but STRONG BUY has 45%, it adjusts thresholds

3. **Learns from mistakes** - If momentum signals fail in choppy markets, it adapts

4. **Improves over time** - Each week's learning cycle refines the model

## Free Data Sources

- **Yahoo Finance** (yfinance) - Price data
- **Reddit JSON API** - Social sentiment
- **Google News RSS** - News headlines

No API keys required!

## Disclaimer

This tool is for **educational purposes only**.

- Trading involves substantial risk of loss
- Past performance doesn't guarantee future results
- Never invest more than you can afford to lose
- Always do your own research
- This is not financial advice

## Contributing

Found a bug or want to improve the learning algorithm? PRs welcome!

---

**The more you use it, the smarter it gets!**
