"use client";

import { Card, Flex, Heading, Text } from "@radix-ui/themes";
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
import type { DashboardModel } from "@/lib/schemas";
import type { MetricMode } from "@/store/dashboard-store";

type ModelPerformanceChartProps = {
  models: DashboardModel[];
  metricMode: MetricMode;
};

export function ModelPerformanceChart({
  models,
  metricMode,
}: ModelPerformanceChartProps) {
  const data = models.map((model) => ({
    model: model.model,
    accuracy: model.accuracyMean,
    macroF1: model.macroF1Mean,
    baseline:
      metricMode === "macro_f1"
        ? model.baselineMacroF1Mean
        : model.baselineAccuracyMean,
    loss: model.lossMean,
  }));

  const metricKey = metricMode === "macro_f1" ? "macroF1" : "accuracy";
  const metricLabel = metricMode === "macro_f1" ? "Macro F1" : "Accuracy";

  return (
    <Card>
      <Flex direction="column" gap="3">
        <Heading size="4">Model Comparison ({metricLabel})</Heading>
        <Text size="2" color="gray">
          Mean test-fold performance vs majority-class baseline.
        </Text>
        <ChartSurface height={320}>
          {({ width, height }) => (
            <BarChart
              width={width}
              height={height}
              data={data}
              margin={{ top: 10, right: 16, bottom: 0, left: 0 }}
            >
              <CartesianGrid strokeDasharray="4 4" strokeOpacity={0.25} />
              <XAxis dataKey="model" />
              <YAxis domain={[0, 1]} />
              <Tooltip formatter={(value) => Number(value).toFixed(4)} />
              <Legend />
              <Bar
                dataKey={metricKey}
                name={metricLabel}
                fill="#0d9488"
                radius={[8, 8, 0, 0]}
              />
              <Bar
                dataKey="baseline"
                name={`Baseline ${metricLabel}`}
                fill="#f59e0b"
                radius={[8, 8, 0, 0]}
              />
            </BarChart>
          )}
        </ChartSurface>
      </Flex>
    </Card>
  );
}
