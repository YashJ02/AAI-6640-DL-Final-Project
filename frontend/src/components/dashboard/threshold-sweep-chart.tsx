"use client";

import { Badge, Callout, Card, Flex, Heading, Text } from "@radix-ui/themes";
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
import type { ModelEvaluation } from "@/lib/schemas";

type ThresholdSweepChartProps = {
  modelName: string;
  evaluation: ModelEvaluation | null | undefined;
};

const EPS = 1e-12;

export function ThresholdSweepChart({
  modelName,
  evaluation,
}: ThresholdSweepChartProps) {
  const raw = evaluation?.thresholdSweep ?? [];

  const baseline = raw.find((point) => point.threshold == null);
  const sweep = raw
    .filter((point) => point.threshold != null)
    .map((point) => ({
      threshold: point.threshold as number,
      sharpe: point.sharpe,
      sortino: point.sortino,
      calmar: point.calmar,
      finalEquity: point.finalEquity,
      maxDrawdown: point.maxDrawdown,
    }));

  const hasSignal = sweep.some(
    (point) =>
      Math.abs(point.sharpe) > EPS ||
      Math.abs(point.sortino) > EPS ||
      Math.abs(point.calmar) > EPS ||
      Math.abs(point.finalEquity - 1.0) > EPS
  );

  const baselineHasSignal =
    baseline != null &&
    (Math.abs(baseline.sharpe) > EPS ||
      Math.abs(baseline.sortino) > EPS ||
      Math.abs(baseline.calmar) > EPS ||
      Math.abs(baseline.finalEquity - 1.0) > EPS);

  return (
    <Card>
      <Flex direction="column" gap="3">
        <Heading size="4">Confidence Threshold Sweep — {modelName}</Heading>
        <Text size="2" color="gray">
          Grid search over MC-Dropout confidence cut-offs. Each point shows
          risk-adjusted performance when the strategy only trades on
          predictions whose mean probability exceeds that threshold.
        </Text>

        {baseline ? (
          <Flex gap="2" wrap="wrap">
            <Badge color="gray" variant="soft">
              Unfiltered baseline — Sharpe {baseline.sharpe.toFixed(2)},
              Sortino {baseline.sortino.toFixed(2)}, Calmar{" "}
              {baseline.calmar.toFixed(2)}, final equity ×{" "}
              {baseline.finalEquity.toFixed(4)}
            </Badge>
            {baselineHasSignal ? (
              <Badge color="green" variant="soft">
                Strategy traded at least once in the unfiltered view
              </Badge>
            ) : (
              <Badge color="amber" variant="soft">
                Unfiltered strategy never entered a position
              </Badge>
            )}
          </Flex>
        ) : null}

        {sweep.length === 0 ? (
          <Text size="2" color="gray">
            No threshold sweep available for this model.
          </Text>
        ) : !hasSignal ? (
          <Callout.Root color="amber" variant="surface">
            <Callout.Text>
              This model&apos;s threshold sweep is flat because the model never
              emitted a confident Up/Down prediction — so the strategy held
              cash at every confidence cut-off. Only models that produce
              non-neutral signals will show meaningful curves here.
            </Callout.Text>
          </Callout.Root>
        ) : (
          <ChartSurface height={320}>
            {({ width, height }) => (
              <LineChart
                width={width}
                height={height}
                data={sweep}
                margin={{ top: 10, right: 16, bottom: 0, left: 0 }}
              >
                <CartesianGrid strokeDasharray="4 4" strokeOpacity={0.25} />
                <XAxis
                  dataKey="threshold"
                  tickFormatter={(value: number) => value.toFixed(2)}
                />
                <YAxis />
                <Tooltip
                  formatter={(value) => Number(value).toFixed(4)}
                  labelFormatter={(label) =>
                    `Threshold ${Number(label).toFixed(2)}`
                  }
                />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="sharpe"
                  stroke="#0d9488"
                  strokeWidth={2}
                  dot={{ r: 3 }}
                  isAnimationActive
                />
                <Line
                  type="monotone"
                  dataKey="sortino"
                  stroke="#f59e0b"
                  strokeWidth={2}
                  dot={{ r: 3 }}
                  isAnimationActive
                />
                <Line
                  type="monotone"
                  dataKey="calmar"
                  stroke="#8b5cf6"
                  strokeWidth={2}
                  dot={{ r: 3 }}
                  isAnimationActive
                />
              </LineChart>
            )}
          </ChartSurface>
        )}
      </Flex>
    </Card>
  );
}
