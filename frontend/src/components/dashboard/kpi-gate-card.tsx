"use client";

import { Badge, Card, Flex, Heading, Table, Text } from "@radix-ui/themes";

import type { KpiReport } from "@/lib/schemas";

type KpiGateCardProps = {
  report: KpiReport | null;
};

function pct(value: number): string {
  return `${(value * 100).toFixed(2)}%`;
}

function signedNumber(value: number, digits = 4): string {
  const sign = value >= 0 ? "+" : "";
  return `${sign}${value.toFixed(digits)}`;
}

export function KpiGateCard({ report }: KpiGateCardProps) {
  if (!report) {
    return (
      <Card>
        <Flex direction="column" gap="2">
          <Heading size="4">KPI Gate</Heading>
          <Text size="2" color="gray">
            No KPI report available.
          </Text>
        </Flex>
      </Card>
    );
  }

  const modelRows = Object.entries(report.models);

  return (
    <Card>
      <Flex direction="column" gap="3">
        <Flex align="center" justify="between" wrap="wrap" gap="2">
          <Heading size="4">KPI Gate</Heading>
          <Flex gap="2" wrap="wrap">
            <Badge color={report.enabled ? "teal" : "gray"} variant="soft">
              Gate {report.enabled ? "enabled" : "advisory"}
            </Badge>
            <Badge color={report.enforce ? "orange" : "gray"} variant="soft">
              Enforce {report.enforce ? "on" : "off"}
            </Badge>
          </Flex>
        </Flex>

        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <Flex direction="column" gap="1">
            <Text size="1" color="gray">
              Accuracy target (min)
            </Text>
            <Heading size="5">{pct(report.target_accuracy_min)}</Heading>
          </Flex>
          <Flex direction="column" gap="1">
            <Text size="1" color="gray">
              Delta-vs-baseline target (min)
            </Text>
            <Heading size="5">
              {signedNumber(report.target_delta_vs_baseline_min)}
            </Heading>
          </Flex>
          <Flex direction="column" gap="1">
            <Text size="1" color="gray">
              Best by accuracy
            </Text>
            <Heading size="5">
              {report.best_by_accuracy?.model ?? "—"}
            </Heading>
            {report.best_by_accuracy ? (
              <Text size="1" color="gray">
                {pct(report.best_by_accuracy.accuracy)}
              </Text>
            ) : null}
          </Flex>
          <Flex direction="column" gap="1">
            <Text size="1" color="gray">
              Best passing model
            </Text>
            <Heading size="5">{report.best_passing_model ?? "—"}</Heading>
            <Text size="1" color="gray">
              {report.any_model_passed
                ? "at least one model passed"
                : "no passing model"}
            </Text>
          </Flex>
        </div>

        <Table.Root size="2" variant="surface">
          <Table.Header>
            <Table.Row>
              <Table.ColumnHeaderCell>Model</Table.ColumnHeaderCell>
              <Table.ColumnHeaderCell>Test Accuracy</Table.ColumnHeaderCell>
              <Table.ColumnHeaderCell>Baseline</Table.ColumnHeaderCell>
              <Table.ColumnHeaderCell>Delta</Table.ColumnHeaderCell>
              <Table.ColumnHeaderCell>Accuracy Gate</Table.ColumnHeaderCell>
              <Table.ColumnHeaderCell>Baseline Gate</Table.ColumnHeaderCell>
              <Table.ColumnHeaderCell>Overall</Table.ColumnHeaderCell>
            </Table.Row>
          </Table.Header>
          <Table.Body>
            {modelRows.map(([name, payload]) => (
              <Table.Row key={name}>
                <Table.RowHeaderCell>
                  <Badge color="teal" variant="soft">
                    {name}
                  </Badge>
                </Table.RowHeaderCell>
                <Table.Cell>{pct(payload.test_accuracy)}</Table.Cell>
                <Table.Cell>{pct(payload.baseline_accuracy)}</Table.Cell>
                <Table.Cell>
                  <Badge
                    color={payload.delta_vs_baseline >= 0 ? "green" : "red"}
                    variant="soft"
                  >
                    {signedNumber(payload.delta_vs_baseline)}
                  </Badge>
                </Table.Cell>
                <Table.Cell>
                  <Badge
                    color={payload.pass_accuracy_target ? "green" : "red"}
                    variant="soft"
                  >
                    {payload.pass_accuracy_target ? "PASS" : "FAIL"}
                  </Badge>
                </Table.Cell>
                <Table.Cell>
                  <Badge
                    color={payload.pass_baseline_gap ? "green" : "red"}
                    variant="soft"
                  >
                    {payload.pass_baseline_gap ? "PASS" : "FAIL"}
                  </Badge>
                </Table.Cell>
                <Table.Cell>
                  <Badge
                    color={payload.pass_all ? "green" : "red"}
                    variant="solid"
                  >
                    {payload.pass_all ? "PASS" : "FAIL"}
                  </Badge>
                </Table.Cell>
              </Table.Row>
            ))}
          </Table.Body>
        </Table.Root>
      </Flex>
    </Card>
  );
}
