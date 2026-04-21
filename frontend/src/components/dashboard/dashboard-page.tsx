"use client";

import { motion } from "motion/react";
import {
  BarChartIcon,
  DashboardIcon,
  LightningBoltIcon,
  QuestionMarkCircledIcon,
} from "@radix-ui/react-icons";
import { Badge, Button, Card, Flex, Heading, Text } from "@radix-ui/themes";
import Link from "next/link";
import { useEffect, useMemo } from "react";

import { useDashboardData } from "@/hooks/use-dashboard-data";
import { useDashboardStore } from "@/store/dashboard-store";

import { BacktestComparison } from "@/components/dashboard/backtest-comparison";
import { BacktestEquityCurve } from "@/components/dashboard/backtest-equity-curve";
import { ConfusionMatrix } from "@/components/dashboard/confusion-matrix";
import { ControlPanel } from "@/components/dashboard/control-panel";
import { DataQualityCard } from "@/components/dashboard/data-quality-card";
import { DecisionBiasesCard } from "@/components/dashboard/decision-biases-card";
import { FeaturePillList } from "@/components/dashboard/feature-pill-list";
import { KeyFindingsCard } from "@/components/dashboard/key-findings-card";
import { KpiGateCard } from "@/components/dashboard/kpi-gate-card";
import { McnemarMatrix } from "@/components/dashboard/mcnemar-matrix";
import { MetricCards } from "@/components/dashboard/metric-cards";
import { ModelPerformanceChart } from "@/components/dashboard/model-performance-chart";
import { OverfitHealthCard } from "@/components/dashboard/overfit-health-card";
import { SharpeMetricsChart } from "@/components/dashboard/sharpe-metrics-chart";
import { ThresholdSweepChart } from "@/components/dashboard/threshold-sweep-chart";
import { TickerReportTable } from "@/components/dashboard/ticker-report-table";
import { TrainingHistoryChart } from "@/components/dashboard/training-history-chart";
import { VolatilityRegimeCard } from "@/components/dashboard/volatility-regime-card";
import { ThemeToggle } from "@/components/theme-toggle";
import { ErrorState } from "@/components/ui/error-state";
import { LoadingState } from "@/components/ui/loading-state";

export function DashboardPage() {
  const { data, isLoading, isError, error, refetch } = useDashboardData();
  const { selectedModel, metricMode, featureSearch, setSelectedModel } =
    useDashboardStore();

  const modelNames = useMemo(
    () => data?.models.map((entry) => entry.model) ?? [],
    [data]
  );

  useEffect(() => {
    if (!selectedModel && modelNames.length > 0) {
      setSelectedModel(modelNames[0]);
    }
  }, [modelNames, selectedModel, setSelectedModel]);

  const activeModel =
    data?.models.find((entry) => entry.model === selectedModel) ??
    data?.models[0] ??
    null;

  const activeEvaluation = activeModel
    ? data?.modelEvaluations[activeModel.model] ?? null
    : null;

  if (isLoading) {
    return <LoadingState />;
  }

  if (isError || !data) {
    return (
      <ErrorState
        message={
          error instanceof Error
            ? error.message
            : "Dashboard data is unavailable."
        }
        onRetry={() => void refetch()}
      />
    );
  }

  const generatedLabel = new Date(data.generatedAt).toLocaleString();

  return (
    <Flex direction="column" gap="5">
      <motion.div
        initial={{ opacity: 0, y: 14 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.45 }}
      >
        <Card className="hero-card">
          <Flex align="center" justify="between" gap="3" wrap="wrap">
            <Flex direction="column" gap="2">
              <Badge color="teal" size="2" variant="surface">
                AAI 6640 Final Project
              </Badge>
              <Heading size="8">
                Intraday Deep Learning Intelligence Console
              </Heading>
              <Text size="3" color="gray">
                LSTM + Temporal Attention · Temporal Fusion Transformer ·
                Dilated CNN-LSTM — a full comparative study on 30 S&amp;P 500
                equities with volatility-normalized labels, walk-forward CV,
                and risk-adjusted backtesting.
              </Text>
              <Flex gap="2" wrap="wrap">
                <Badge color="teal" variant="soft">
                  <DashboardIcon />
                  {data.models.length} models tracked
                </Badge>
                <Badge color="orange" variant="soft">
                  <BarChartIcon />
                  {data.featureColumns.length} engineered features
                </Badge>
                <Badge color="purple" variant="soft">
                  <LightningBoltIcon />
                  generated {generatedLabel}
                </Badge>
              </Flex>
            </Flex>
            <Flex gap="2" align="center">
              <Link href="/help" prefetch>
                <Button variant="soft" color="teal" size="3">
                  <QuestionMarkCircledIcon />
                  Help &amp; Guide
                </Button>
              </Link>
              <ThemeToggle />
            </Flex>
          </Flex>
        </Card>
      </motion.div>

      <KeyFindingsCard payload={data} />

      <ControlPanel models={modelNames} />

      <MetricCards model={activeModel} />

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        <DataQualityCard summary={data.dataQualitySummary} />
        <DecisionBiasesCard model={activeModel} />
      </div>

      <KpiGateCard report={data.kpiReport} />

      <OverfitHealthCard report={data.overfitReport} />

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <ModelPerformanceChart models={data.models} metricMode={metricMode} />
        <FeaturePillList features={data.featureColumns} query={featureSearch} />
      </div>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        <TrainingHistoryChart
          modelName={activeModel?.model ?? "n/a"}
          history={
            activeModel ? data.trainingHistory[activeModel.model] ?? [] : []
          }
        />
        <ConfusionMatrix
          matrix={
            activeModel?.confusionMatrix ?? [
              [0, 0, 0],
              [0, 0, 0],
              [0, 0, 0],
            ]
          }
          modelName={activeModel?.model ?? "n/a"}
        />
      </div>

      <BacktestEquityCurve
        modelName={activeModel?.model ?? "n/a"}
        evaluation={activeEvaluation}
      />

      <BacktestComparison payload={data} />

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        <ThresholdSweepChart
          modelName={activeModel?.model ?? "n/a"}
          evaluation={activeEvaluation}
        />
        <VolatilityRegimeCard
          modelName={activeModel?.model ?? "n/a"}
          evaluation={activeEvaluation}
        />
      </div>

      <SharpeMetricsChart sharpeSummary={data.sharpeSummary} />

      <McnemarMatrix mcnemar={data.mcnemar} />

      <TickerReportTable
        cleaning={data.tickerCleaningReport}
        modeling={data.tickerModelingReport}
      />
    </Flex>
  );
}
