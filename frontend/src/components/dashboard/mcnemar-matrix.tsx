"use client";

import { Badge, Card, Flex, Heading, Table, Text } from "@radix-ui/themes";

import type { DashboardPayload } from "@/lib/schemas";

type McnemarMatrixProps = {
  mcnemar: DashboardPayload["mcnemar"];
};

function parsePair(key: string): [string, string] | null {
  const parts = key.split("_vs_");
  if (parts.length !== 2) {
    return null;
  }
  return [parts[0], parts[1]];
}

function pValueBadge(pValue: number) {
  if (pValue < 0.01) {
    return <Badge color="green">p &lt; 0.01 (highly significant)</Badge>;
  }
  if (pValue < 0.05) {
    return <Badge color="teal">p &lt; 0.05 (significant)</Badge>;
  }
  if (pValue < 0.1) {
    return <Badge color="amber">p &lt; 0.1 (marginal)</Badge>;
  }
  return (
    <Badge color="gray" variant="soft">
      not significant
    </Badge>
  );
}

export function McnemarMatrix({ mcnemar }: McnemarMatrixProps) {
  const rows = Object.entries(mcnemar).flatMap(([pairKey, entries]) => {
    const parsed = parsePair(pairKey);
    if (!parsed) return [];
    const [a, b] = parsed;
    return entries.map((entry) => ({ pairKey, a, b, entry }));
  });

  return (
    <Card>
      <Flex direction="column" gap="3">
        <Heading size="4">McNemar Pairwise Significance</Heading>
        <Text size="2" color="gray">
          Chi-squared test on discordant predictions between every pair of
          models, per walk-forward fold. A low p-value means the two models
          disagree on statistically-different examples.
        </Text>

        {rows.length === 0 ? (
          <Text size="2" color="gray">
            No McNemar statistics available.
          </Text>
        ) : (
          <Table.Root size="2" variant="surface">
            <Table.Header>
              <Table.Row>
                <Table.ColumnHeaderCell>Model A</Table.ColumnHeaderCell>
                <Table.ColumnHeaderCell>Model B</Table.ColumnHeaderCell>
                <Table.ColumnHeaderCell>Fold</Table.ColumnHeaderCell>
                <Table.ColumnHeaderCell>n01 (only A wrong)</Table.ColumnHeaderCell>
                <Table.ColumnHeaderCell>n10 (only B wrong)</Table.ColumnHeaderCell>
                <Table.ColumnHeaderCell>χ²</Table.ColumnHeaderCell>
                <Table.ColumnHeaderCell>p-value</Table.ColumnHeaderCell>
                <Table.ColumnHeaderCell>Verdict</Table.ColumnHeaderCell>
              </Table.Row>
            </Table.Header>
            <Table.Body>
              {rows.map(({ pairKey, a, b, entry }) => (
                <Table.Row key={`${pairKey}-${entry.foldId}`}>
                  <Table.RowHeaderCell>
                    <Badge color="teal" variant="soft">
                      {a}
                    </Badge>
                  </Table.RowHeaderCell>
                  <Table.Cell>
                    <Badge color="orange" variant="soft">
                      {b}
                    </Badge>
                  </Table.Cell>
                  <Table.Cell>{entry.foldId}</Table.Cell>
                  <Table.Cell>{entry.n01.toFixed(0)}</Table.Cell>
                  <Table.Cell>{entry.n10.toFixed(0)}</Table.Cell>
                  <Table.Cell>{entry.chi2.toFixed(3)}</Table.Cell>
                  <Table.Cell>{entry.pValue.toFixed(4)}</Table.Cell>
                  <Table.Cell>{pValueBadge(entry.pValue)}</Table.Cell>
                </Table.Row>
              ))}
            </Table.Body>
          </Table.Root>
        )}
      </Flex>
    </Card>
  );
}
