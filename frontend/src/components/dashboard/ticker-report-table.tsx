"use client";

import {
  Badge,
  Card,
  Flex,
  Heading,
  ScrollArea,
  SegmentedControl,
  Table,
  Text,
} from "@radix-ui/themes";
import { useState } from "react";

import type {
  TickerCleaningRow,
  TickerModelingRow,
} from "@/lib/schemas";

type TickerReportTableProps = {
  cleaning: TickerCleaningRow[];
  modeling: TickerModelingRow[];
};

type TabValue = "modeling" | "cleaning";

function pct(value: number): string {
  return `${(value * 100).toFixed(2)}%`;
}

function fmtInt(value: number): string {
  return new Intl.NumberFormat().format(value);
}

export function TickerReportTable({
  cleaning,
  modeling,
}: TickerReportTableProps) {
  const [tab, setTab] = useState<TabValue>("modeling");

  return (
    <Card>
      <Flex direction="column" gap="3">
        <Flex align="center" justify="between" wrap="wrap" gap="2">
          <Heading size="4">Per-Ticker Data Reports</Heading>
          <SegmentedControl.Root
            value={tab}
            onValueChange={(value) => setTab(value as TabValue)}
          >
            <SegmentedControl.Item value="modeling">
              Modeling
            </SegmentedControl.Item>
            <SegmentedControl.Item value="cleaning">
              Cleaning
            </SegmentedControl.Item>
          </SegmentedControl.Root>
        </Flex>

        {tab === "modeling" ? (
          modeling.length === 0 ? (
            <Text size="2" color="gray">
              No modeling report available.
            </Text>
          ) : (
            <ScrollArea
              type="auto"
              scrollbars="both"
              style={{ maxHeight: 380 }}
            >
              <Table.Root size="1" variant="surface">
                <Table.Header>
                  <Table.Row>
                    <Table.ColumnHeaderCell>Ticker</Table.ColumnHeaderCell>
                    <Table.ColumnHeaderCell>
                      After Cleaning
                    </Table.ColumnHeaderCell>
                    <Table.ColumnHeaderCell>
                      After Features
                    </Table.ColumnHeaderCell>
                    <Table.ColumnHeaderCell>
                      After Labels
                    </Table.ColumnHeaderCell>
                    <Table.ColumnHeaderCell>Down %</Table.ColumnHeaderCell>
                    <Table.ColumnHeaderCell>Neutral %</Table.ColumnHeaderCell>
                    <Table.ColumnHeaderCell>Up %</Table.ColumnHeaderCell>
                  </Table.Row>
                </Table.Header>
                <Table.Body>
                  {modeling.map((row) => (
                    <Table.Row key={row.ticker}>
                      <Table.RowHeaderCell>
                        <Badge color="teal" variant="soft">
                          {row.ticker}
                        </Badge>
                      </Table.RowHeaderCell>
                      <Table.Cell>{fmtInt(row.rowsAfterCleaning)}</Table.Cell>
                      <Table.Cell>{fmtInt(row.rowsAfterFeatures)}</Table.Cell>
                      <Table.Cell>{fmtInt(row.rowsAfterLabels)}</Table.Cell>
                      <Table.Cell>{pct(row.label0Pct)}</Table.Cell>
                      <Table.Cell>{pct(row.label1Pct)}</Table.Cell>
                      <Table.Cell>{pct(row.label2Pct)}</Table.Cell>
                    </Table.Row>
                  ))}
                </Table.Body>
              </Table.Root>
            </ScrollArea>
          )
        ) : cleaning.length === 0 ? (
          <Text size="2" color="gray">
            No cleaning report available.
          </Text>
        ) : (
          <ScrollArea type="auto" scrollbars="both" style={{ maxHeight: 380 }}>
            <Table.Root size="1" variant="surface">
              <Table.Header>
                <Table.Row>
                  <Table.ColumnHeaderCell>Ticker</Table.ColumnHeaderCell>
                  <Table.ColumnHeaderCell>Rows Before</Table.ColumnHeaderCell>
                  <Table.ColumnHeaderCell>Rows After</Table.ColumnHeaderCell>
                  <Table.ColumnHeaderCell>Dropped</Table.ColumnHeaderCell>
                  <Table.ColumnHeaderCell>Drop %</Table.ColumnHeaderCell>
                  <Table.ColumnHeaderCell>Dup TS</Table.ColumnHeaderCell>
                  <Table.ColumnHeaderCell>Non-session</Table.ColumnHeaderCell>
                  <Table.ColumnHeaderCell>
                    Invalid OHLCV
                  </Table.ColumnHeaderCell>
                  <Table.ColumnHeaderCell>
                    Outlier Return
                  </Table.ColumnHeaderCell>
                  <Table.ColumnHeaderCell>
                    Outlier Range
                  </Table.ColumnHeaderCell>
                  <Table.ColumnHeaderCell>
                    Low Coverage
                  </Table.ColumnHeaderCell>
                  <Table.ColumnHeaderCell>
                    Kept Sessions
                  </Table.ColumnHeaderCell>
                </Table.Row>
              </Table.Header>
              <Table.Body>
                {cleaning.map((row) => (
                  <Table.Row key={row.ticker}>
                    <Table.RowHeaderCell>
                      <Badge color="teal" variant="soft">
                        {row.ticker}
                      </Badge>
                    </Table.RowHeaderCell>
                    <Table.Cell>{fmtInt(row.rowsBefore)}</Table.Cell>
                    <Table.Cell>{fmtInt(row.rowsAfter)}</Table.Cell>
                    <Table.Cell>{fmtInt(row.droppedRows)}</Table.Cell>
                    <Table.Cell>{row.dropPct.toFixed(2)}%</Table.Cell>
                    <Table.Cell>{row.droppedDuplicateTs}</Table.Cell>
                    <Table.Cell>{row.droppedNonSession}</Table.Cell>
                    <Table.Cell>{row.droppedInvalidOhlcv}</Table.Cell>
                    <Table.Cell>{row.droppedOutlierReturn}</Table.Cell>
                    <Table.Cell>{row.droppedOutlierRange}</Table.Cell>
                    <Table.Cell>{row.droppedLowCoverageSession}</Table.Cell>
                    <Table.Cell>{row.keptSessions}</Table.Cell>
                  </Table.Row>
                ))}
              </Table.Body>
            </Table.Root>
          </ScrollArea>
        )}
      </Flex>
    </Card>
  );
}
