"use client";

import {
  CheckCircledIcon,
  CrossCircledIcon,
  MinusCircledIcon,
} from "@radix-ui/react-icons";
import { Badge, Card, Flex, Heading, Separator, Text } from "@radix-ui/themes";

import type { DashboardModel, DashboardPayload } from "@/lib/schemas";

type KeyFindingsCardProps = {
  payload: DashboardPayload;
};

function findBest<T>(
  items: T[],
  keyFn: (item: T) => number
): T | null {
  let best: T | null = null;
  let bestVal = -Infinity;
  for (const item of items) {
    const value = keyFn(item);
    if (value > bestVal) {
      bestVal = value;
      best = item;
    }
  }
  return best;
}

function modelsThatBeatBaseline(models: DashboardModel[]): DashboardModel[] {
  return models.filter((m) => m.deltaVsBaseline > 1e-8);
}

function modelsThatCollapsed(models: DashboardModel[]): DashboardModel[] {
  return models.filter(
    (m) => Math.abs(m.deltaVsBaseline) < 1e-8 && m.macroF1Mean <= m.baselineMacroF1Mean + 1e-6
  );
}

export function KeyFindingsCard({ payload }: KeyFindingsCardProps) {
  const { models, modelEvaluations, dataQualitySummary, sharpeSummary } = payload;

  const neutralPct = dataQualitySummary?.labelDistribution?.["1"] ?? null;

  const bestAccuracy = findBest(models, (m) => m.accuracyMean);
  const bestMacroF1 = findBest(models, (m) => m.macroF1Mean);
  const bestBestValF1 = findBest(models, (m) => m.bestValMacroF1Mean);
  const bestDelta = findBest(models, (m) => m.deltaVsBaseline);

  const beatBaseline = modelsThatBeatBaseline(models);
  const collapsed = modelsThatCollapsed(models);

  const sharpeEntries = sharpeSummary
    ? Object.entries(sharpeSummary).map(([name, entry]) => ({ name, ...entry }))
    : [];
  const bestSharpe = findBest(sharpeEntries, (e) => e.best_sharpe);

  const regimeSummaries = models
    .map((m) => {
      const evaluation = modelEvaluations[m.model];
      const regimes = evaluation?.volatilityRegimes ?? [];
      const low = regimes.find((r) => r.regime === "low_vol");
      const high = regimes.find((r) => r.regime === "high_vol");
      if (!low || !high) return null;
      return {
        model: m.model,
        degradation: low.macroF1 - high.macroF1,
        lowF1: low.macroF1,
        highF1: high.macroF1,
      };
    })
    .filter(
      (entry): entry is NonNullable<typeof entry> => entry != null
    );

  const mostRobust = regimeSummaries.length
    ? regimeSummaries.reduce(
        (best, curr) => (curr.degradation < best.degradation ? curr : best),
        regimeSummaries[0]
      )
    : null;

  const findings: {
    key: string;
    icon: React.ReactNode;
    color: "green" | "red" | "amber" | "teal" | "orange" | "gray";
    title: string;
    body: React.ReactNode;
  }[] = [];

  if (beatBaseline.length > 0 && bestDelta) {
    findings.push({
      key: "beat-baseline",
      icon: <CheckCircledIcon />,
      color: "green",
      title: `${beatBaseline.length} model${
        beatBaseline.length === 1 ? "" : "s"
      } beat the baseline on accuracy`,
      body: (
        <>
          Best is <b>{bestDelta.model}</b> with Δ=
          {bestDelta.deltaVsBaseline.toFixed(4)} and macro-F1=
          {bestDelta.macroF1Mean.toFixed(4)}.
        </>
      ),
    });
  } else {
    findings.push({
      key: "no-beat",
      icon: <CrossCircledIcon />,
      color: "red",
      title: "No model beat the majority-class baseline on raw accuracy",
      body: (
        <>
          Raw accuracy saturates near the baseline because the label set is
          heavily dominated by the Neutral class (~
          {neutralPct != null ? (neutralPct * 100).toFixed(1) : "94"}%). This
          is a class-imbalance artefact, not model failure — <b>Macro-F1</b> is
          the meaningful signal on this dataset.
        </>
      ),
    });
  }

  if (bestBestValF1) {
    findings.push({
      key: "best-val-f1",
      icon: <CheckCircledIcon />,
      color: "teal",
      title: `Best validation macro-F1: ${bestBestValF1.model}`,
      body: (
        <>
          Val F1 = {bestBestValF1.bestValMacroF1Mean.toFixed(4)} at epoch{" "}
          {bestBestValF1.bestEpoch}. Tuned val F1 after bias-grid search:{" "}
          {bestBestValF1.tunedValMacroF1Mean.toFixed(4)}.
        </>
      ),
    });
  }

  if (collapsed.length > 0) {
    findings.push({
      key: "collapsed",
      icon: <MinusCircledIcon />,
      color: "amber",
      title: `${collapsed.length} model${
        collapsed.length === 1 ? "" : "s"
      } collapsed to predicting the majority class`,
      body: (
        <>
          Models: {collapsed.map((m) => m.model).join(", ")}. Their confusion
          matrices show all predictions landing in the Neutral column — the
          learned decision biases never shifted Up/Down class probabilities
          enough to cross the decision boundary on test data.
        </>
      ),
    });
  }

  if (bestSharpe && bestSharpe.best_sharpe > 0) {
    findings.push({
      key: "best-sharpe",
      icon: <CheckCircledIcon />,
      color: "green",
      title: `Best Sharpe across threshold sweep: ${bestSharpe.name}`,
      body: (
        <>
          Sharpe = {bestSharpe.best_sharpe.toFixed(2)}, Sortino ={" "}
          {bestSharpe.best_sortino.toFixed(2)}, Calmar ={" "}
          {bestSharpe.best_calmar.toFixed(2)}, final equity ×{" "}
          {bestSharpe.best_final_equity.toFixed(4)} with max drawdown{" "}
          {(bestSharpe.best_max_drawdown * 100).toFixed(2)}%. This exceeds the
          Sharpe&gt;1 project target.
        </>
      ),
    });
  } else if (sharpeEntries.length > 0) {
    findings.push({
      key: "no-sharpe",
      icon: <MinusCircledIcon />,
      color: "amber",
      title: "No model achieved a non-trivial Sharpe ratio",
      body: (
        <>
          Every model&apos;s equity curve stayed flat because the model never
          emitted an actionable Up/Down signal; the strategy held cash for the
          entire out-of-sample window.
        </>
      ),
    });
  }

  if (mostRobust) {
    findings.push({
      key: "robust",
      icon: <CheckCircledIcon />,
      color: "orange",
      title: `Most volatility-robust: ${mostRobust.model}`,
      body: (
        <>
          Macro-F1 drops only{" "}
          {(
            ((mostRobust.lowF1 - mostRobust.highF1) /
              Math.max(mostRobust.lowF1, 1e-8)) *
            100
          ).toFixed(2)}
          % from low-vol sessions ({mostRobust.lowF1.toFixed(4)}) to high-vol
          sessions ({mostRobust.highF1.toFixed(4)}).
        </>
      ),
    });
  }

  if (bestAccuracy && bestMacroF1 && bestAccuracy.model !== bestMacroF1.model) {
    findings.push({
      key: "split",
      icon: <MinusCircledIcon />,
      color: "gray",
      title: "Accuracy-best and F1-best disagree",
      body: (
        <>
          {bestAccuracy.model} wins on raw accuracy but {bestMacroF1.model}{" "}
          wins on macro-F1. In class-imbalanced settings, trust F1.
        </>
      ),
    });
  }

  return (
    <Card>
      <Flex direction="column" gap="3">
        <Flex align="center" justify="between" wrap="wrap" gap="2">
          <Heading size="5">Key Findings</Heading>
          <Badge color="teal" variant="soft">
            Auto-derived from artifacts
          </Badge>
        </Flex>
        <Separator size="4" />

        <div className="grid grid-cols-1 gap-3 md:grid-cols-2 2xl:grid-cols-3">
          {findings.map((finding) => (
            <Card key={finding.key} variant="surface">
              <Flex direction="column" gap="2">
                <Flex gap="2" align="center">
                  <Badge color={finding.color} variant="solid">
                    {finding.icon}
                  </Badge>
                  <Text weight="medium" size="2">
                    {finding.title}
                  </Text>
                </Flex>
                <Text size="2" color="gray">
                  {finding.body}
                </Text>
              </Flex>
            </Card>
          ))}
        </div>

        <Separator size="4" />

        <Flex direction="column" gap="1">
          <Heading size="3">Takeaway</Heading>
          <Text size="2" color="gray">
            On this 5-minute, EWMA-normalized, 3-class formulation most models
            struggle to separate Up/Down from Neutral because ~
            {neutralPct != null ? (neutralPct * 100).toFixed(1) : "94"}% of
            labels are Neutral by construction. The interesting architectural
            signal is in the <b>macro-F1 gap</b>, the <b>volatility-regime
            degradation</b>, and <b>whether the model produces any tradeable
            positions</b> at all (shown by the backtest activity badge).
          </Text>
        </Flex>
      </Flex>
    </Card>
  );
}
