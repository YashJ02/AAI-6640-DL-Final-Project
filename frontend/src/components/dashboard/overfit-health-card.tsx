"use client";

import { Badge, Card, Flex, Heading, Table, Text } from "@radix-ui/themes";

import type { OverfitReport } from "@/lib/schemas";

type OverfitHealthCardProps = {
  report: OverfitReport | null;
};

function gapColor(gap: number): "green" | "amber" | "red" | "gray" {
  if (!Number.isFinite(gap)) return "gray";
  const absGap = Math.abs(gap);
  if (absGap < 0.02) return "green";
  if (absGap < 0.05) return "amber";
  return "red";
}

function fmt(value: number, digits = 4): string {
  return Number.isFinite(value) ? value.toFixed(digits) : "—";
}

export function OverfitHealthCard({ report }: OverfitHealthCardProps) {
  const entries = report ? Object.entries(report) : [];

  return (
    <Card>
      <Flex direction="column" gap="3">
        <Heading size="4">Overfitting Health</Heading>
        <Text size="2" color="gray">
          Generalization gap = validation macro-F1 minus training macro-F1. A
          small positive or zero gap indicates healthy learning; a large
          negative gap signals overfitting.
        </Text>
        {entries.length === 0 ? (
          <Text size="2" color="gray">
            No overfit health report available.
          </Text>
        ) : (
          <Table.Root size="2" variant="surface">
            <Table.Header>
              <Table.Row>
                <Table.ColumnHeaderCell>Model</Table.ColumnHeaderCell>
                <Table.ColumnHeaderCell>Epochs</Table.ColumnHeaderCell>
                <Table.ColumnHeaderCell>Best Epoch</Table.ColumnHeaderCell>
                <Table.ColumnHeaderCell>Best Val F1</Table.ColumnHeaderCell>
                <Table.ColumnHeaderCell>Train F1 @ Best</Table.ColumnHeaderCell>
                <Table.ColumnHeaderCell>Best-Epoch Gap</Table.ColumnHeaderCell>
                <Table.ColumnHeaderCell>Final Train F1</Table.ColumnHeaderCell>
                <Table.ColumnHeaderCell>Final Val F1</Table.ColumnHeaderCell>
                <Table.ColumnHeaderCell>Final Gap</Table.ColumnHeaderCell>
              </Table.Row>
            </Table.Header>
            <Table.Body>
              {entries.map(([name, payload]) => (
                <Table.Row key={name}>
                  <Table.RowHeaderCell>
                    <Badge color="teal" variant="soft">
                      {name}
                    </Badge>
                  </Table.RowHeaderCell>
                  <Table.Cell>{payload.epochs_ran}</Table.Cell>
                  <Table.Cell>{payload.best_epoch_by_val_f1}</Table.Cell>
                  <Table.Cell>{fmt(payload.best_val_macro_f1)}</Table.Cell>
                  <Table.Cell>
                    {fmt(payload.best_train_macro_f1_same_epoch)}
                  </Table.Cell>
                  <Table.Cell>
                    <Badge
                      color={gapColor(payload.best_epoch_gap_val_minus_train)}
                      variant="soft"
                    >
                      {fmt(payload.best_epoch_gap_val_minus_train)}
                    </Badge>
                  </Table.Cell>
                  <Table.Cell>
                    {fmt(payload.last_epoch_train_macro_f1)}
                  </Table.Cell>
                  <Table.Cell>
                    {fmt(payload.last_epoch_val_macro_f1)}
                  </Table.Cell>
                  <Table.Cell>
                    <Badge
                      color={gapColor(payload.last_epoch_gap_val_minus_train)}
                      variant="soft"
                    >
                      {fmt(payload.last_epoch_gap_val_minus_train)}
                    </Badge>
                  </Table.Cell>
                </Table.Row>
              ))}
            </Table.Body>
          </Table.Root>
        )}
      </Flex>
    </Card>
  );
}
