"use client";

import { Badge, Card, Flex, Heading, Text } from "@radix-ui/themes";

import type { DashboardModel } from "@/lib/schemas";

type DecisionBiasesCardProps = {
  model: DashboardModel | null;
};

const CLASS_LABELS = ["Down", "Neutral", "Up"];
const CLASS_COLORS: Array<"red" | "gray" | "green"> = ["red", "gray", "green"];

function fmtSigned(value: number, digits = 3): string {
  const sign = value >= 0 ? "+" : "";
  return `${sign}${value.toFixed(digits)}`;
}

export function DecisionBiasesCard({ model }: DecisionBiasesCardProps) {
  if (!model) {
    return null;
  }

  return (
    <Card>
      <Flex direction="column" gap="3">
        <Heading size="4">Decision Bias Tuning — {model.model}</Heading>
        <Text size="2" color="gray">
          Per-class logit shifts selected on the validation split after training
          to maximize macro-F1. A negative bias on Down/Up lowers the bar for
          those predictions; Neutral=0 keeps that class unchanged.
        </Text>

        <div className="grid grid-cols-3 gap-3">
          {model.decisionBiasesMean.map((value, index) => (
            <Flex key={CLASS_LABELS[index]} direction="column" gap="1">
              <Text size="1" color="gray">
                {CLASS_LABELS[index]}
              </Text>
              <Heading size="5">
                <Badge color={CLASS_COLORS[index]} variant="soft">
                  {fmtSigned(value)}
                </Badge>
              </Heading>
            </Flex>
          ))}
        </div>

        <Flex gap="2" wrap="wrap">
          <Badge color="teal" variant="soft">
            Best val macro-F1: {model.bestValMacroF1Mean.toFixed(4)}
          </Badge>
          <Badge color="orange" variant="soft">
            Tuned val macro-F1: {model.tunedValMacroF1Mean.toFixed(4)}
          </Badge>
          <Badge color="gray" variant="soft">
            Total folds: {model.totalFolds}
          </Badge>
        </Flex>
      </Flex>
    </Card>
  );
}
