"use client";

import { Badge, Card, Flex, Heading, Text } from "@radix-ui/themes";
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
import type { DashboardPayload } from "@/lib/schemas";

type BacktestComparisonProps = {
  payload: DashboardPayload;
};

const MODEL_COLORS = ["#0d9488", "#ea580c", "#8b5cf6", "#2563eb", "#db2777"];

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

export function BacktestComparison({ payload }: BacktestComparisonProps) {
  const seriesList = payload.models
    .map((model) => {
      const evaluation = payload.modelEvaluations[model.model];
      const curve = evaluation?.backtestUnfiltered?.curve ?? [];
      if (curve.length === 0) {
        return null;
      }
      return {
        model: model.model,
        activity:
          evaluation?.backtestUnfiltered?.strategyActivityRatio ?? 0,
        points: curve.map((point) => ({
          timestamp: point.timestamp,
          [model.model]: point.equityCurve,
        })),
      };
    })
    .filter((entry): entry is NonNullable<typeof entry> => entry != null);

  if (seriesList.length === 0) {
    return null;
  }

  const benchmark =
    payload.modelEvaluations[seriesList[0].model]?.backtestUnfiltered?.curve ??
    [];

  const byTimestamp = new Map<string, Record<string, number | string>>();

  for (const series of seriesList) {
    for (const point of series.points) {
      const existing = byTimestamp.get(point.timestamp) ?? {
        timestamp: point.timestamp,
      };
      byTimestamp.set(point.timestamp, { ...existing, ...point });
    }
  }

  for (const benchmarkPoint of benchmark) {
    const existing = byTimestamp.get(benchmarkPoint.timestamp) ?? {
      timestamp: benchmarkPoint.timestamp,
    };
    existing.benchmark = benchmarkPoint.benchmarkCurve;
    byTimestamp.set(benchmarkPoint.timestamp, existing);
  }

  const merged = [...byTimestamp.values()].sort((a, b) =>
    String(a.timestamp).localeCompare(String(b.timestamp))
  );

  return (
    <Card>
      <Flex direction="column" gap="3">
        <Flex align="center" justify="between" wrap="wrap" gap="2">
          <Flex direction="column" gap="1">
            <Heading size="4">Cross-Model Equity Comparison (Unfiltered)</Heading>
            <Text size="2" color="gray">
              Every model&apos;s strategy equity curve plotted against the same
              buy-and-hold benchmark. Models with a flat line at 1.0 never
              emitted an actionable prediction in this window.
            </Text>
          </Flex>
          <Flex gap="2" wrap="wrap">
            {seriesList.map((s) => (
              <Badge
                key={s.model}
                color={s.activity > 0 ? "green" : "gray"}
                variant="soft"
              >
                {s.model} · {(s.activity * 100).toFixed(1)}% active
              </Badge>
            ))}
          </Flex>
        </Flex>

        <ChartSurface height={340}>
          {({ width, height }) => (
            <LineChart
              width={width}
              height={height}
              data={merged}
              margin={{ top: 10, right: 16, bottom: 0, left: 0 }}
            >
              <CartesianGrid strokeDasharray="4 4" strokeOpacity={0.25} />
              <XAxis
                dataKey="timestamp"
                tickFormatter={(value) => tickTimestamp(String(value))}
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
              {seriesList.map((series, index) => (
                <Line
                  key={series.model}
                  type="monotone"
                  dataKey={series.model}
                  name={`${series.model} equity`}
                  stroke={MODEL_COLORS[index % MODEL_COLORS.length]}
                  strokeWidth={2}
                  dot={false}
                  isAnimationActive
                />
              ))}
              <Line
                type="monotone"
                dataKey="benchmark"
                name="Buy-and-hold benchmark"
                stroke="#f59e0b"
                strokeDasharray="6 4"
                strokeWidth={2}
                dot={false}
                isAnimationActive
              />
            </LineChart>
          )}
        </ChartSurface>
      </Flex>
    </Card>
  );
}
