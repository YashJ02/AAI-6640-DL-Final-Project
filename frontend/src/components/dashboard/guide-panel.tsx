"use client";

import {
  InfoCircledIcon,
  LightningBoltIcon,
  MagnifyingGlassIcon,
} from "@radix-ui/react-icons";
import {
  Badge,
  Callout,
  Card,
  Flex,
  Heading,
  Separator,
  Text,
} from "@radix-ui/themes";

type ResearchQuestion = {
  id: string;
  question: string;
  dashboardSection: string;
  badges: string[];
};

const RESEARCH_QUESTIONS: ResearchQuestion[] = [
  {
    id: "RQ1",
    question:
      "Can deep learning beat a naive majority-class baseline at 5-minute direction prediction on S&P 500 intraday data?",
    dashboardSection:
      "Metric cards (Test Accuracy vs Baseline Accuracy) + Model Comparison bar chart. Look for a positive Δ vs baseline.",
    badges: ["baseline", "accuracy", "macro F1"],
  },
  {
    id: "RQ2",
    question:
      "Among LSTM + Attention, Temporal Fusion Transformer, and Dilated CNN-LSTM — which architecture generalizes best?",
    dashboardSection:
      "Model Comparison chart, Training Dynamics (per-model tabs), Confusion Matrix, and McNemar pairwise significance.",
    badges: ["LSTM", "TFT", "CNN-LSTM", "ensemble"],
  },
  {
    id: "RQ3",
    question:
      "Which of the 28+ engineered features (OHLCV log-returns, 18 technicals, Fourier time, related-market context) carry the most predictive signal?",
    dashboardSection:
      "Feature Space panel lists every engineered feature; the backend TFT-VSN / mutual-information / ablation rankings write CSVs when `run_feature_importance_pipeline` executes.",
    badges: ["feature engineering", "VSN", "mutual information"],
  },
  {
    id: "RQ4",
    question:
      "How much does accuracy degrade in high-volatility regimes vs calm sessions?",
    dashboardSection:
      "Volatility Regime Breakdown per model, with explicit degradation badges (high-vol − low-vol).",
    badges: ["ATR regimes", "degradation %"],
  },
  {
    id: "RQ5",
    question:
      "Do model signals produce a positive risk-adjusted return (Sharpe > 1) in out-of-sample backtesting, with MC-Dropout confidence filtering?",
    dashboardSection:
      "Backtest equity curve (Unfiltered vs MC-Dropout Filtered), Confidence Threshold Sweep, and Sharpe Optimization Summary.",
    badges: ["Sharpe", "Sortino", "Calmar", "MC Dropout"],
  },
];

type GuidePanelProps = {
  neutralClassPct: number | null;
};

export function GuidePanel({ neutralClassPct }: GuidePanelProps) {
  return (
    <Card>
      <Flex direction="column" gap="4">
        <Flex align="center" justify="between" wrap="wrap" gap="2">
          <Flex direction="column" gap="1">
            <Heading size="5">How to read this dashboard</Heading>
            <Text size="2" color="gray">
              An overview of the project, what every section means, and how to
              draw conclusions.
            </Text>
          </Flex>
          <Flex gap="2" wrap="wrap">
            <Badge color="teal" variant="soft">
              <InfoCircledIcon /> Reader-first
            </Badge>
            <Badge color="orange" variant="soft">
              <MagnifyingGlassIcon /> Research-driven
            </Badge>
          </Flex>
        </Flex>

        <Separator size="4" />

        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <Flex direction="column" gap="2">
            <Heading size="4">The problem</Heading>
            <Text size="2" color="gray">
              Intraday stock prices are noisy, non-stationary, and heavy-tailed.
              Classical ARIMA models fail at them. This study asks whether
              three modern deep-learning architectures — an attention-LSTM, a
              Temporal Fusion Transformer, and a dilated CNN-LSTM hybrid — can
              learn useful 5-minute direction signals across a 30-equity S&amp;P
              500 universe and whether those signals translate into
              risk-adjusted trading performance.
            </Text>
          </Flex>

          <Flex direction="column" gap="2">
            <Heading size="4">The data</Heading>
            <Text size="2" color="gray">
              5-minute bars via <b>yfinance</b>, 30 tickers across 5 sectors,
              ~2 months of history per ticker. Features: log-returns,
              high-low range, body/shadow, volume log-change, 18 technical
              indicators (RSI, MACD, Bollinger, EMA-9/21, ATR, VWAP, OBV, ADX,
              Stoch %K/%D, CCI, Williams %R, MFI) plus Fourier time-of-day
              features and related-market context (SPY, QQQ, VIX, IWM, DIA,
              TLT, GLD, DXY).
            </Text>
          </Flex>

          <Flex direction="column" gap="2">
            <Heading size="4">The label</Heading>
            <Text size="2" color="gray">
              3-class direction: <b>Down / Neutral / Up</b>. Returns are first
              z-scored against <b>EWMA volatility</b> (RiskMetrics λ = 0.94),
              then thresholded at ±0.5σ. A move is therefore classified
              relative to how calm or turbulent the market is <i>right now</i>
              , not in absolute percentage terms.
            </Text>
          </Flex>

          <Flex direction="column" gap="2">
            <Heading size="4">How it&apos;s validated</Heading>
            <Text size="2" color="gray">
              <b>Walk-forward cross-validation</b> over session IDs (no future
              leakage). Focal loss + label smoothing. Decision biases are
              tuned on validation to maximize macro-F1. McNemar&apos;s test
              compares each pair of models on the same held-out bars for
              statistical significance.
            </Text>
          </Flex>
        </div>

        <Separator size="4" />

        <Flex direction="column" gap="3">
          <Heading size="4">Research questions &amp; where to find them</Heading>
          <div className="grid grid-cols-1 gap-3 xl:grid-cols-2">
            {RESEARCH_QUESTIONS.map((rq) => (
              <Card key={rq.id} variant="surface">
                <Flex direction="column" gap="2">
                  <Flex gap="2" align="center" wrap="wrap">
                    <Badge color="teal" variant="solid">
                      {rq.id}
                    </Badge>
                    {rq.badges.map((badge) => (
                      <Badge key={badge} color="gray" variant="soft">
                        {badge}
                      </Badge>
                    ))}
                  </Flex>
                  <Text size="2" weight="medium">
                    {rq.question}
                  </Text>
                  <Text size="1" color="gray">
                    <b>Where:</b> {rq.dashboardSection}
                  </Text>
                </Flex>
              </Card>
            ))}
          </div>
        </Flex>

        <Separator size="4" />

        <Flex direction="column" gap="3">
          <Heading size="4">How to draw a conclusion</Heading>
          <ol className="list-decimal pl-5 text-[13px] leading-6 text-[var(--gray-11)] space-y-1">
            <li>
              <b>Start with the Key Findings card</b> — it summarizes the
              numbers the rest of the dashboard breaks down.
            </li>
            <li>
              <b>Look at Δ vs Baseline</b> in the metric cards. Because
              ~{neutralClassPct != null ? (neutralClassPct * 100).toFixed(0) : "94"}%
              of labels are <i>neutral</i>, a model that predicts &quot;always
              neutral&quot; scores ~{neutralClassPct != null
                ? (neutralClassPct * 100).toFixed(0)
                : "94"}% accuracy. Raw accuracy alone is therefore
              misleading; the signal is in <b>Macro-F1</b> and the <b>Δ vs
              baseline</b>.
            </li>
            <li>
              <b>Check the confusion matrix</b>. If all predictions fall in
              the Neutral column, the model collapsed — it hasn&apos;t learned
              a decision boundary yet.
            </li>
            <li>
              <b>Cross-check in McNemar</b> — p-values below 0.05 mean two
              models disagree on statistically different sets of bars.
            </li>
            <li>
              <b>Read the backtest</b>. Sharpe &gt; 1 is the practical bar set
              in the project proposal. Compare Strategy vs Buy-&amp;-Hold
              benchmark on the same bars.
            </li>
            <li>
              <b>Check volatility regime degradation</b>. A robust model
              shouldn&apos;t lose more than a few F1 percentage points between
              calm and turbulent sessions.
            </li>
          </ol>
        </Flex>

        <Callout.Root color="amber" variant="surface">
          <Callout.Icon>
            <LightningBoltIcon />
          </Callout.Icon>
          <Callout.Text>
            <b>Important context.</b> With only ~2 months of 5-minute bars per
            ticker (provider lookback limit), walk-forward folds are shallow.
            Treat numbers as indicative of methodology rather than production
            trading signals.
          </Callout.Text>
        </Callout.Root>
      </Flex>
    </Card>
  );
}
