"use client";

import { Badge, Callout, Card, Flex, Heading, Table, Text } from "@radix-ui/themes";
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
import type { DashboardPayload } from "@/lib/schemas";

type SharpeMetricsChartProps = {
  sharpeSummary: DashboardPayload["sharpeSummary"];
};

const EPS = 1e-9;

type Entry = DashboardPayload["sharpeSummary"] extends infer S
  ? S extends Record<string, infer V>
    ? V
    : never
  : never;

function fmt(value: number | null | undefined, digits = 4): string {
  if (value == null || !Number.isFinite(value)) {
    return "—";
  }
  return value.toFixed(digits);
}

function isInactive(entry: Entry): boolean {
  return (
    Math.abs(entry.best_sharpe) < EPS &&
    Math.abs(entry.best_sortino) < EPS &&
    Math.abs(entry.best_calmar) < EPS &&
    Math.abs(entry.best_final_equity - 1.0) < EPS &&
    Math.abs(entry.best_max_drawdown) < EPS
  );
}

export function SharpeMetricsChart({
  sharpeSummary,
}: SharpeMetricsChartProps) {
  if (!sharpeSummary || Object.keys(sharpeSummary).length === 0) {
    return (
      <Card>
        <Flex direction="column" gap="2">
          <Heading size="4">Sharpe Optimization Summary</Heading>
          <Text size="2" color="gray">
            No sharpe optimization artifact available.
          </Text>
        </Flex>
      </Card>
    );
  }

  const entries = Object.entries(sharpeSummary).map(([model, entry]) => ({
    model,
    entry,
    inactive: isInactive(entry),
  }));

  const activeEntries = entries.filter((row) => !row.inactive);
  const inactiveEntries = entries.filter((row) => row.inactive);

  const chartData = activeEntries.map(({ model, entry }) => ({
    model,
    Sharpe: entry.best_sharpe,
    Sortino: entry.best_sortino,
    Calmar: entry.best_calmar,
  }));

  return (
    <Card>
      <Flex direction="column" gap="3">
        <Heading size="4">Sharpe Optimization Summary (best threshold)</Heading>
        <Text size="2" color="gray">
          Per-model best risk-adjusted returns obtained over the confidence
          threshold sweep.
        </Text>

        {inactiveEntries.length > 0 ? (
          <Callout.Root color="amber" variant="surface">
            <Callout.Text>
              <b>
                {inactiveEntries.map((row) => row.model).join(", ")}
              </b>{" "}
              never traded — their strategy held cash for every bar. The
              backend&apos;s risk module returns 0 for Sharpe/Sortino/Calmar
              when the strategy return stream has zero variance (see{" "}
              <code>compute_risk_metrics</code>). Those rows are shown below
              for completeness but are excluded from the bar chart because
              0-Sharpe there would misrepresent &quot;no activity&quot; as
              &quot;neutral performance&quot;.
            </Callout.Text>
          </Callout.Root>
        ) : null}

        {chartData.length === 0 ? (
          <Text size="2" color="gray">
            No active strategies to chart.
          </Text>
        ) : (
          <ChartSurface height={300}>
            {({ width, height }) => (
              <BarChart
                width={width}
                height={height}
                data={chartData}
                margin={{ top: 10, right: 16, bottom: 0, left: 0 }}
              >
                <CartesianGrid strokeDasharray="4 4" strokeOpacity={0.25} />
                <XAxis dataKey="model" />
                <YAxis />
                <Tooltip formatter={(value) => Number(value).toFixed(4)} />
                <Legend />
                <Bar dataKey="Sharpe" fill="#0d9488" radius={[8, 8, 0, 0]} />
                <Bar dataKey="Sortino" fill="#f59e0b" radius={[8, 8, 0, 0]} />
                <Bar dataKey="Calmar" fill="#8b5cf6" radius={[8, 8, 0, 0]} />
              </BarChart>
            )}
          </ChartSurface>
        )}

        <Table.Root size="2" variant="surface">
          <Table.Header>
            <Table.Row>
              <Table.ColumnHeaderCell>Model</Table.ColumnHeaderCell>
              <Table.ColumnHeaderCell>Activity</Table.ColumnHeaderCell>
              <Table.ColumnHeaderCell>Best Threshold</Table.ColumnHeaderCell>
              <Table.ColumnHeaderCell>Sharpe</Table.ColumnHeaderCell>
              <Table.ColumnHeaderCell>Sortino</Table.ColumnHeaderCell>
              <Table.ColumnHeaderCell>Calmar</Table.ColumnHeaderCell>
              <Table.ColumnHeaderCell>Final Equity</Table.ColumnHeaderCell>
              <Table.ColumnHeaderCell>Max Drawdown</Table.ColumnHeaderCell>
            </Table.Row>
          </Table.Header>
          <Table.Body>
            {entries.map(({ model, entry, inactive }) => (
              <Table.Row key={model}>
                <Table.RowHeaderCell>{model}</Table.RowHeaderCell>
                <Table.Cell>
                  {inactive ? (
                    <Badge color="amber" variant="soft">
                      strategy never traded
                    </Badge>
                  ) : (
                    <Badge color="green" variant="soft">
                      active
                    </Badge>
                  )}
                </Table.Cell>
                <Table.Cell>
                  {entry.best_threshold == null
                    ? "unfiltered"
                    : entry.best_threshold.toFixed(2)}
                </Table.Cell>
                <Table.Cell>
                  {inactive ? (
                    <Text color="gray">—</Text>
                  ) : (
                    fmt(entry.best_sharpe, 3)
                  )}
                </Table.Cell>
                <Table.Cell>
                  {inactive ? (
                    <Text color="gray">—</Text>
                  ) : (
                    fmt(entry.best_sortino, 3)
                  )}
                </Table.Cell>
                <Table.Cell>
                  {inactive ? (
                    <Text color="gray">—</Text>
                  ) : (
                    fmt(entry.best_calmar, 3)
                  )}
                </Table.Cell>
                <Table.Cell>
                  {fmt(entry.best_final_equity, 4)}
                </Table.Cell>
                <Table.Cell>
                  {inactive ? (
                    <Text color="gray">—</Text>
                  ) : (
                    `${(entry.best_max_drawdown * 100).toFixed(2)}%`
                  )}
                </Table.Cell>
              </Table.Row>
            ))}
          </Table.Body>
        </Table.Root>
      </Flex>
    </Card>
  );
}
