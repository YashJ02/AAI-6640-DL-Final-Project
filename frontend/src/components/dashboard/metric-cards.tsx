"use client";

import { motion } from "motion/react";
import { Badge, Card, Flex, Heading, Text } from "@radix-ui/themes";

import type { DashboardModel } from "@/lib/schemas";

type MetricCardsProps = {
  model: DashboardModel | null;
};

function formatPct(value: number, digits = 2) {
  return `${(value * 100).toFixed(digits)}%`;
}

function signed(value: number, digits = 4): string {
  const sign = value >= 0 ? "+" : "";
  return `${sign}${value.toFixed(digits)}`;
}

const EPS = 1e-8;

function trendBadge(
  value: number,
  label: string
): { color: "green" | "red" | "gray"; text: string } | null {
  if (Math.abs(value) < EPS) {
    return { color: "gray", text: `= baseline (${label})` };
  }
  return {
    color: value > 0 ? "green" : "red",
    text: `${signed(value)} ${label}`,
  };
}

export function MetricCards({ model }: MetricCardsProps) {
  if (!model) {
    return null;
  }

  const accuracyTrend = trendBadge(model.deltaVsBaseline, "vs baseline");
  const f1Trend = trendBadge(model.deltaF1VsBaseline, "vs baseline");

  const metrics: {
    label: string;
    value: string;
    subtitle: string;
    badge?: { color: "green" | "red" | "gray" | "teal" | "orange"; text: string } | null;
  }[] = [
    {
      label: "Test Accuracy",
      value: formatPct(model.accuracyMean),
      subtitle:
        model.accuracyStd > 0
          ? `± ${model.accuracyStd.toFixed(4)} across ${model.totalFolds} folds`
          : `single-fold result`,
      badge: accuracyTrend,
    },
    {
      label: "Test Macro F1",
      value: model.macroF1Mean.toFixed(4),
      subtitle:
        model.macroF1Std > 0
          ? `± ${model.macroF1Std.toFixed(4)} across ${model.totalFolds} folds`
          : `single-fold result`,
      badge: f1Trend,
    },
    {
      label: "Test Loss",
      value: model.lossMean.toFixed(4),
      subtitle:
        model.lossStd > 0
          ? `± ${model.lossStd.toFixed(4)}`
          : "cross-entropy / focal",
    },
    {
      label: "Baseline Accuracy",
      value: formatPct(model.baselineAccuracyMean),
      subtitle: "predict majority class every bar",
      badge: { color: "orange", text: "reference" },
    },
    {
      label: "Baseline Macro F1",
      value: model.baselineMacroF1Mean.toFixed(4),
      subtitle: "majority-class macro-F1",
      badge: { color: "orange", text: "reference" },
    },
    {
      label: "Best Val Macro F1",
      value: model.bestValMacroF1Mean.toFixed(4),
      subtitle: `tuned val F1: ${model.tunedValMacroF1Mean.toFixed(4)}`,
      badge:
        model.bestValMacroF1Mean > model.macroF1Mean + EPS
          ? { color: "teal", text: "val > test" }
          : null,
    },
    {
      label: "Best Epoch",
      value: model.bestEpoch.toString(),
      subtitle: `${model.totalFolds} walk-forward fold${model.totalFolds === 1 ? "" : "s"}`,
    },
    {
      label: "Decision Biases",
      value: model.decisionBiasesMean
        .map((value) => signed(value, 2))
        .join(" / "),
      subtitle: "Down / Neutral / Up logit shifts",
    },
  ];

  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
      {metrics.map((item, index) => (
        <motion.div
          key={item.label}
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.35, delay: index * 0.04 }}
        >
          <Card>
            <Flex direction="column" gap="2">
              <Text size="2" color="gray">
                {item.label}
              </Text>
              <Heading size="6">{item.value}</Heading>
              <Text size="1" color="gray">
                {item.subtitle}
              </Text>
              {item.badge ? (
                <Badge color={item.badge.color} variant="soft">
                  {item.badge.text}
                </Badge>
              ) : null}
            </Flex>
          </Card>
        </motion.div>
      ))}
    </div>
  );
}
