"use client";

import { Card, Flex, Heading, SegmentedControl, Text } from "@radix-ui/themes";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { TrainingPoint } from "@/lib/schemas";
import { ChartSurface } from "@/components/ui/chart-surface";
import {
  useDashboardStore,
  type TrainingMetric,
} from "@/store/dashboard-store";

type TrainingHistoryChartProps = {
  modelName: string;
  history: TrainingPoint[];
};

type SeriesConfig = {
  domain: [number | string, number | string];
  lines: {
    dataKey: keyof TrainingPoint;
    name: string;
    stroke: string;
  }[];
  tooltipFormatter?: (value: number) => string;
};

function scientific(value: number): string {
  if (!Number.isFinite(value)) {
    return "0";
  }
  if (value === 0) {
    return "0";
  }
  return value.toExponential(2);
}

const SERIES: Record<TrainingMetric, SeriesConfig> = {
  f1: {
    domain: [0, 1],
    lines: [
      { dataKey: "trainMacroF1", name: "Train Macro F1", stroke: "#0f766e" },
      { dataKey: "valMacroF1", name: "Validation Macro F1", stroke: "#f59e0b" },
    ],
    tooltipFormatter: (value) => value.toFixed(4),
  },
  loss: {
    domain: ["auto", "auto"],
    lines: [
      { dataKey: "trainLoss", name: "Train Loss", stroke: "#0f766e" },
      { dataKey: "valLoss", name: "Validation Loss", stroke: "#f59e0b" },
    ],
    tooltipFormatter: (value) => value.toFixed(4),
  },
  learningRate: {
    domain: [0, "auto"],
    lines: [
      {
        dataKey: "learningRate",
        name: "Learning Rate",
        stroke: "#8b5cf6",
      },
    ],
    tooltipFormatter: scientific,
  },
};

export function TrainingHistoryChart({
  modelName,
  history,
}: TrainingHistoryChartProps) {
  const { trainingMetric, setTrainingMetric } = useDashboardStore();
  const config = SERIES[trainingMetric];

  return (
    <Card>
      <Flex direction="column" gap="3">
        <Flex align="center" justify="between" wrap="wrap" gap="2">
          <Heading size="4">Training Dynamics — {modelName}</Heading>
          <SegmentedControl.Root
            value={trainingMetric}
            onValueChange={(value) =>
              setTrainingMetric(value as TrainingMetric)
            }
          >
            <SegmentedControl.Item value="f1">Macro F1</SegmentedControl.Item>
            <SegmentedControl.Item value="loss">Loss</SegmentedControl.Item>
            <SegmentedControl.Item value="learningRate">
              LR
            </SegmentedControl.Item>
          </SegmentedControl.Root>
        </Flex>
        {history.length === 0 ? (
          <Text size="2" color="gray">
            No training history found for this model.
          </Text>
        ) : (
          <ChartSurface height={320}>
            {({ width, height }) => (
              <LineChart
                width={width}
                height={height}
                data={history}
                margin={{ top: 10, right: 16, bottom: 0, left: 0 }}
              >
                <CartesianGrid strokeDasharray="4 4" strokeOpacity={0.25} />
                <XAxis dataKey="epoch" />
                <YAxis
                  domain={config.domain}
                  tickFormatter={
                    trainingMetric === "learningRate" ? scientific : undefined
                  }
                />
                <Tooltip
                  formatter={(value) =>
                    config.tooltipFormatter
                      ? config.tooltipFormatter(Number(value))
                      : Number(value).toFixed(4)
                  }
                />
                <Legend />
                {config.lines.map((line) => (
                  <Line
                    key={line.dataKey}
                    type="monotone"
                    dataKey={line.dataKey as string}
                    name={line.name}
                    stroke={line.stroke}
                    strokeWidth={2}
                    dot={false}
                    isAnimationActive
                  />
                ))}
              </LineChart>
            )}
          </ChartSurface>
        )}
      </Flex>
    </Card>
  );
}
