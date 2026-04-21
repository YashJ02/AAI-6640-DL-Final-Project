"use client";

import {
  Badge,
  Callout,
  Card,
  Flex,
  Heading,
  SegmentedControl,
  Separator,
  Text,
} from "@radix-ui/themes";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { ChartSurface } from "@/components/ui/chart-surface";
import type { BacktestView, ModelEvaluation } from "@/lib/schemas";
import {
  useDashboardStore,
  type BacktestMode,
} from "@/store/dashboard-store";

type BacktestEquityCurveProps = {
  modelName: string;
  evaluation: ModelEvaluation | null | undefined;
};

function tickTimestamp(value: string): string {
  try {
    const parsed = new Date(value);
    if (Number.isNaN(parsed.getTime())) {
      return value;
    }
    return parsed.toLocaleString(undefined, {
      month: "short",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return value;
  }
}

function fmtFixed(value: number | undefined, digits = 4): string {
  if (value == null || !Number.isFinite(value)) {
    return "—";
  }
  return value.toFixed(digits);
}

function fmtPct(value: number | undefined, digits = 2): string {
  if (value == null || !Number.isFinite(value)) {
    return "—";
  }
  return `${(value * 100).toFixed(digits)}%`;
}

function MetricsRow({ view }: { view: BacktestView }) {
  const strategy = view.strategyMetrics;
  const benchmark = view.benchmarkMetrics;

  const cells: {
    label: string;
    strategy: string;
    benchmark: string;
  }[] = [
    {
      label: "Annualized Return",
      strategy: fmtPct(strategy.annualizedReturn),
      benchmark: fmtPct(benchmark.annualizedReturn),
    },
    {
      label: "Annualized Volatility",
      strategy: fmtPct(strategy.annualizedVolatility),
      benchmark: fmtPct(benchmark.annualizedVolatility),
    },
    {
      label: "Sharpe",
      strategy: fmtFixed(strategy.sharpe, 3),
      benchmark: fmtFixed(benchmark.sharpe, 3),
    },
    {
      label: "Sortino",
      strategy: fmtFixed(strategy.sortino, 3),
      benchmark: fmtFixed(benchmark.sortino, 3),
    },
    {
      label: "Calmar",
      strategy: fmtFixed(strategy.calmar, 3),
      benchmark: fmtFixed(benchmark.calmar, 3),
    },
    {
      label: "Max Drawdown",
      strategy: fmtPct(strategy.maxDrawdown, 2),
      benchmark: fmtPct(benchmark.maxDrawdown, 2),
    },
    {
      label: "Final Equity",
      strategy: fmtFixed(strategy.finalEquity, 4),
      benchmark: fmtFixed(benchmark.finalEquity, 4),
    },
    {
      label: "DD Duration (bars)",
      strategy: String(strategy.maxDrawdownDurationBars ?? "—"),
      benchmark: String(benchmark.maxDrawdownDurationBars ?? "—"),
    },
  ];

  return (
    <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
      {cells.map((cell) => (
        <Flex key={cell.label} direction="column" gap="1">
          <Text size="1" color="gray">
            {cell.label}
          </Text>
          <Flex direction="column" gap="1">
            <Text size="3" weight="medium">
              <Badge color="teal" variant="soft" mr="1">
                Strategy
              </Badge>
              {cell.strategy}
            </Text>
            <Text size="2" color="gray">
              <Badge color="orange" variant="soft" mr="1">
                Benchmark
              </Badge>
              {cell.benchmark}
            </Text>
          </Flex>
        </Flex>
      ))}
    </div>
  );
}

export function BacktestEquityCurve({
  modelName,
  evaluation,
}: BacktestEquityCurveProps) {
  const { backtestMode, setBacktestMode } = useDashboardStore();

  const view =
    backtestMode === "filtered"
      ? evaluation?.backtestFiltered ?? null
      : evaluation?.backtestUnfiltered ?? null;

  const hasActivity = view != null && view.strategyActivityRatio > 0;

  return (
    <Card>
      <Flex direction="column" gap="3">
        <Flex
          align={{ initial: "start", sm: "center" }}
          justify="between"
          wrap="wrap"
          gap="2"
        >
          <Flex direction="column" gap="1">
            <Heading size="4">Backtest — {modelName}</Heading>
            <Text size="2" color="gray">
              Strategy equity curve vs buy-and-hold benchmark on held-out
              session bars. &quot;Up&quot; predictions enter/hold long,
              &quot;Down&quot; exits to cash, &quot;Neutral&quot; keeps the
              prior position.
            </Text>
          </Flex>
          <SegmentedControl.Root
            value={backtestMode}
            onValueChange={(value) => setBacktestMode(value as BacktestMode)}
          >
            <SegmentedControl.Item value="unfiltered">
              Unfiltered
            </SegmentedControl.Item>
            <SegmentedControl.Item value="filtered">
              MC-Dropout Filtered
            </SegmentedControl.Item>
          </SegmentedControl.Root>
        </Flex>

        <Separator size="4" />

        {view == null ? (
          <Text size="2" color="gray">
            No backtest artifact for this view.
          </Text>
        ) : (
          <Flex direction="column" gap="3">
            <MetricsRow view={view} />

            <Flex gap="2" wrap="wrap">
              <Badge
                color={hasActivity ? "green" : "amber"}
                variant="soft"
              >
                Strategy active on{" "}
                {(view.strategyActivityRatio * 100).toFixed(2)}% of bars
              </Badge>
              {hasActivity ? null : (
                <Badge color="gray" variant="soft">
                  equity stays at 1.0 by design
                </Badge>
              )}
            </Flex>

            {!hasActivity ? (
              <Callout.Root color="amber" variant="surface">
                <Callout.Text>
                  <b>No trades were executed in this view.</b> The model&apos;s
                  predictions never produced an Up or Down signal (they all
                  landed on Neutral), so the strategy held cash for the entire
                  30-day out-of-sample window. The benchmark curve still
                  tracks the equal-weight buy-and-hold of the ticker basket.
                </Callout.Text>
              </Callout.Root>
            ) : null}

            {view.curve.length === 0 ? (
              <Text size="2" color="gray">
                No equity curve data points.
              </Text>
            ) : (
              <ChartSurface height={320}>
                {({ width, height }) => (
                  <LineChart
                    width={width}
                    height={height}
                    data={view.curve}
                    margin={{ top: 10, right: 16, bottom: 0, left: 0 }}
                  >
                    <CartesianGrid strokeDasharray="4 4" strokeOpacity={0.25} />
                    <XAxis
                      dataKey="timestamp"
                      tickFormatter={tickTimestamp}
                      minTickGap={40}
                    />
                    <YAxis
                      domain={["auto", "auto"]}
                      tickFormatter={(value: number) => value.toFixed(3)}
                    />
                    <Tooltip
                      labelFormatter={(label) => tickTimestamp(String(label))}
                      formatter={(value) => Number(value).toFixed(5)}
                    />
                    <Legend />
                    <Line
                      type="monotone"
                      dataKey="equityCurve"
                      name="Strategy Equity"
                      stroke="#0d9488"
                      strokeWidth={2}
                      dot={false}
                      isAnimationActive
                    />
                    <Line
                      type="monotone"
                      dataKey="benchmarkCurve"
                      name="Benchmark (Buy & Hold)"
                      stroke="#f59e0b"
                      strokeWidth={2}
                      dot={false}
                      isAnimationActive
                    />
                  </LineChart>
                )}
              </ChartSurface>
            )}
          </Flex>
        )}
      </Flex>
    </Card>
  );
}
