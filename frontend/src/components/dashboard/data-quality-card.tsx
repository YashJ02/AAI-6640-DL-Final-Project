"use client";

import {
  Badge,
  Card,
  Flex,
  Heading,
  Separator,
  Text,
} from "@radix-ui/themes";
import { Cell, Legend, Pie, PieChart, Tooltip } from "recharts";

import { ChartSurface } from "@/components/ui/chart-surface";
import type { DataQualitySummary } from "@/lib/schemas";

type DataQualityCardProps = {
  summary: DataQualitySummary | null;
};

const CLASS_LABELS: Record<string, string> = {
  "0": "Down",
  "1": "Neutral",
  "2": "Up",
};

const CLASS_COLORS: Record<string, string> = {
  "0": "#ef4444",
  "1": "#64748b",
  "2": "#10b981",
};

function formatTimestamp(value: string | null): string {
  if (!value) {
    return "—";
  }

  try {
    const parsed = new Date(value);
    if (Number.isNaN(parsed.getTime())) {
      return value;
    }

    return parsed.toLocaleString(undefined, {
      year: "numeric",
      month: "short",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return value;
  }
}

function formatInt(value: number): string {
  return new Intl.NumberFormat().format(value);
}

export function DataQualityCard({ summary }: DataQualityCardProps) {
  if (!summary) {
    return (
      <Card>
        <Flex direction="column" gap="2">
          <Heading size="4">Data Quality Summary</Heading>
          <Text size="2" color="gray">
            No data-quality summary artifact found.
          </Text>
        </Flex>
      </Card>
    );
  }

  const distribution = Object.entries(summary.labelDistribution)
    .map(([key, value]) => ({
      key,
      name: CLASS_LABELS[key] ?? `Class ${key}`,
      value,
    }))
    .sort((a, b) => a.key.localeCompare(b.key));

  return (
    <Card>
      <Flex direction="column" gap="3">
        <Flex align="center" justify="between" wrap="wrap" gap="2">
          <Heading size="4">Data Quality Summary</Heading>
          <Flex gap="2" wrap="wrap">
            <Badge color="teal" variant="soft">
              {formatInt(summary.featureCount)} total features
            </Badge>
            <Badge color="orange" variant="soft">
              {formatInt(summary.featureCount - summary.relatedFeatureCount)} engineered
            </Badge>
            <Badge color="purple" variant="soft">
              {formatInt(summary.relatedFeatureCount)} related-market
            </Badge>
          </Flex>
        </Flex>

        <Separator size="4" />

        <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
          <Flex direction="column" gap="1">
            <Text size="1" color="gray">
              Rows in modeling table
            </Text>
            <Heading size="5">{formatInt(summary.rows)}</Heading>
          </Flex>
          <Flex direction="column" gap="1">
            <Text size="1" color="gray">
              Tickers (equities)
            </Text>
            <Heading size="5">{formatInt(summary.tickers)}</Heading>
          </Flex>
          <Flex direction="column" gap="1">
            <Text size="1" color="gray">
              Total features
            </Text>
            <Heading size="5">{formatInt(summary.featureCount)}</Heading>
          </Flex>
          <Flex direction="column" gap="1">
            <Text size="1" color="gray">
              Related-market features
            </Text>
            <Heading size="5">{formatInt(summary.relatedFeatureCount)}</Heading>
          </Flex>
        </div>

        <Separator size="4" />

        <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
          <Flex direction="column" gap="1">
            <Text size="1" color="gray">
              Coverage window start
            </Text>
            <Text size="3" weight="medium">
              {formatTimestamp(summary.timestampMin)}
            </Text>
          </Flex>
          <Flex direction="column" gap="1">
            <Text size="1" color="gray">
              Coverage window end
            </Text>
            <Text size="3" weight="medium">
              {formatTimestamp(summary.timestampMax)}
            </Text>
          </Flex>
        </div>

        <Separator size="4" />

        <Flex direction="column" gap="2">
          <Text weight="medium" size="2">
            Label distribution (volatility-normalized EWMA)
          </Text>
          {distribution.length === 0 ? (
            <Text size="2" color="gray">
              Label distribution unavailable.
            </Text>
          ) : (
            <ChartSurface height={240}>
              {({ width, height }) => (
                <PieChart width={width} height={height}>
                  <Pie
                    data={distribution}
                    dataKey="value"
                    nameKey="name"
                    cx="50%"
                    cy="50%"
                    innerRadius={55}
                    outerRadius={90}
                    paddingAngle={2}
                    isAnimationActive
                  >
                    {distribution.map((entry) => (
                      <Cell
                        key={entry.key}
                        fill={CLASS_COLORS[entry.key] ?? "#14b8a6"}
                      />
                    ))}
                  </Pie>
                  <Tooltip
                    formatter={(value) =>
                      `${(Number(value) * 100).toFixed(2)}%`
                    }
                  />
                  <Legend />
                </PieChart>
              )}
            </ChartSurface>
          )}
        </Flex>
      </Flex>
    </Card>
  );
}
