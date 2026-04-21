"use client";

import { Badge, Card, Flex, Heading, Text } from "@radix-ui/themes";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { ChartSurface } from "@/components/ui/chart-surface";
import type { ModelEvaluation } from "@/lib/schemas";

type VolatilityRegimeCardProps = {
  modelName: string;
  evaluation: ModelEvaluation | null | undefined;
};

function formatInt(value: number): string {
  return new Intl.NumberFormat().format(value);
}

function sign(value: number): string {
  return value >= 0 ? "+" : "";
}

export function VolatilityRegimeCard({
  modelName,
  evaluation,
}: VolatilityRegimeCardProps) {
  const regimes = evaluation?.volatilityRegimes ?? [];
  const degradation = evaluation?.degradation;

  const data = regimes.map((row) => ({
    regime: row.regime,
    accuracy: row.accuracy,
    macroF1: row.macroF1,
    count: row.count,
  }));

  return (
    <Card>
      <Flex direction="column" gap="3">
        <Heading size="4">
          Volatility Regime Breakdown — {modelName}
        </Heading>
        <Text size="2" color="gray">
          Test sessions split by ATR-percentile into high-vol vs low-vol.
          Degradation shows how much performance drops in the high-vol regime.
        </Text>

        {degradation ? (
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            <Flex direction="column" gap="1">
              <Text size="1" color="gray">
                Accuracy degradation (high vs low vol)
              </Text>
              <Heading size="5">
                <Badge
                  color={
                    degradation.accuracyDegradationPct > 0 ? "red" : "green"
                  }
                  variant="soft"
                >
                  {sign(degradation.accuracyDegradationPct)}
                  {degradation.accuracyDegradationPct.toFixed(3)}%
                </Badge>
              </Heading>
            </Flex>
            <Flex direction="column" gap="1">
              <Text size="1" color="gray">
                Macro F1 degradation (high vs low vol)
              </Text>
              <Heading size="5">
                <Badge
                  color={
                    degradation.macroF1DegradationPct > 0 ? "red" : "green"
                  }
                  variant="soft"
                >
                  {sign(degradation.macroF1DegradationPct)}
                  {degradation.macroF1DegradationPct.toFixed(3)}%
                </Badge>
              </Heading>
            </Flex>
          </div>
        ) : null}

        {data.length === 0 ? (
          <Text size="2" color="gray">
            No volatility regime metrics available.
          </Text>
        ) : (
          <>
            <ChartSurface height={300}>
              {({ width, height }) => (
                <BarChart
                  width={width}
                  height={height}
                  data={data}
                  margin={{ top: 10, right: 16, bottom: 0, left: 0 }}
                >
                  <CartesianGrid strokeDasharray="4 4" strokeOpacity={0.25} />
                  <XAxis dataKey="regime" />
                  <YAxis domain={[0, 1]} />
                  <Tooltip
                    formatter={(value) => Number(value).toFixed(4)}
                  />
                  <Legend />
                  <Bar
                    dataKey="accuracy"
                    name="Accuracy"
                    fill="#0d9488"
                    radius={[8, 8, 0, 0]}
                  />
                  <Bar
                    dataKey="macroF1"
                    name="Macro F1"
                    fill="#f59e0b"
                    radius={[8, 8, 0, 0]}
                  />
                </BarChart>
              )}
            </ChartSurface>

            <Flex gap="3" wrap="wrap">
              {data.map((row) => (
                <Badge key={row.regime} color="teal" variant="soft">
                  {row.regime}: {formatInt(row.count)} samples
                </Badge>
              ))}
            </Flex>
          </>
        )}
      </Flex>
    </Card>
  );
}
