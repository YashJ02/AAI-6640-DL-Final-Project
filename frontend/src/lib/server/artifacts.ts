import { promises as fs } from "node:fs";
import path from "node:path";

import Papa from "papaparse";

import {
  DataQualitySummarySchema,
  DashboardPayloadSchema,
  KpiAccuracyReportSchema,
  ModelEvaluationSchema,
  OverfitHealthReportSchema,
  ResultsSummaryRawSchema,
  SharpeSummarySchema,
  TickerCleaningRowSchema,
  TickerModelingRowSchema,
  type DashboardPayload,
  type TickerCleaningRow,
  type TickerModelingRow,
  type TrainingPoint,
} from "@/lib/schemas";

const PROJECT_ROOT = path.resolve(process.cwd(), "..");
const ARTIFACTS_DIR = path.join(PROJECT_ROOT, "artifacts");

const RESULTS_SUMMARY_PATH = path.join(ARTIFACTS_DIR, "results_summary.json");
const KPI_REPORT_PATH = path.join(ARTIFACTS_DIR, "kpi_accuracy_report.json");
const OVERFIT_REPORT_PATH = path.join(
  ARTIFACTS_DIR,
  "overfit_health_report.json"
);
const DATA_QUALITY_SUMMARY_PATH = path.join(
  ARTIFACTS_DIR,
  "data_quality",
  "modeling_summary.json"
);
const TICKER_MODELING_REPORT_PATH = path.join(
  ARTIFACTS_DIR,
  "data_quality",
  "ticker_modeling_report.csv"
);
const TICKER_CLEANING_REPORT_PATH = path.join(
  ARTIFACTS_DIR,
  "data_quality",
  "ticker_cleaning_report.csv"
);
const EVALUATION_DIR = path.join(ARTIFACTS_DIR, "evaluation_accuracy90");
const SHARPE_SUMMARY_PATH = path.join(
  ARTIFACTS_DIR,
  "evaluation_accuracy90",
  "sharpe_optimization_summary.json"
);
const TRAINING_LOGS_DIR = path.join(ARTIFACTS_DIR, "training_logs");

type CsvRow = {
  epoch: string;
  train_loss: string;
  train_macro_f1: string;
  val_loss: string;
  val_macro_f1: string;
  learning_rate: string;
};

type ThresholdGridRow = {
  threshold: string;
  sharpe: string;
  sortino: string;
  calmar: string;
  final_equity: string;
  max_drawdown: string;
};

type RegimeRow = {
  regime: string;
  count: string;
  accuracy: string;
  macro_f1: string;
};

type CurveRow = {
  timestamp: string;
  strategy_return: string;
  equity_curve: string;
  benchmark_curve: string;
};

type TickerModelingCsvRow = {
  ticker: string;
  rows_after_cleaning: string;
  rows_after_features: string;
  rows_after_labels: string;
  label_0_pct: string;
  label_1_pct: string;
  label_2_pct: string;
};

type TickerCleaningCsvRow = {
  ticker: string;
  rows_before: string;
  rows_after: string;
  dropped_rows: string;
  drop_pct: string;
  dropped_duplicate_ts: string;
  dropped_non_session: string;
  dropped_invalid_ohlcv: string;
  dropped_outlier_return: string;
  dropped_outlier_range: string;
  dropped_low_coverage_session: string;
  kept_sessions: string;
};

async function readJsonFile<T>(filePath: string): Promise<T | null> {
  try {
    const raw = await fs.readFile(filePath, "utf-8");
    return JSON.parse(raw) as T;
  } catch {
    return null;
  }
}

async function readCsvRows<T>(filePath: string): Promise<T[]> {
  try {
    const csvRaw = await fs.readFile(filePath, "utf-8");
    const parsed = Papa.parse<T>(csvRaw, {
      header: true,
      skipEmptyLines: true,
    });
    return parsed.data;
  } catch {
    return [];
  }
}

function averageConfusionMatrix(matrices: number[][][]): number[][] {
  if (matrices.length === 0) {
    return [
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
    ];
  }

  const rows = matrices[0]?.length ?? 0;
  const cols = matrices[0]?.[0]?.length ?? 0;
  const output = Array.from({ length: rows }, () =>
    Array<number>(cols).fill(0)
  );

  for (const matrix of matrices) {
    for (let r = 0; r < rows; r += 1) {
      for (let c = 0; c < cols; c += 1) {
        output[r][c] += matrix[r][c] / matrices.length;
      }
    }
  }

  return output;
}

function toNumber(value: string | undefined): number {
  const parsed = Number(value ?? "0");
  return Number.isFinite(parsed) ? parsed : 0;
}

function toNullableNumber(value: string | undefined): number | null {
  if (value == null || value.trim() === "") {
    return null;
  }

  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function mean(values: number[]): number {
  if (values.length === 0) {
    return 0;
  }

  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function downsampleCurve<T>(rows: T[], maxPoints = 320): T[] {
  if (rows.length <= maxPoints) {
    return rows;
  }

  const sampled: T[] = [];
  const lastIndex = rows.length - 1;
  const stride = lastIndex / (maxPoints - 1);

  for (let idx = 0; idx < maxPoints; idx += 1) {
    sampled.push(rows[Math.round(idx * stride)]);
  }

  return sampled;
}

function normalizeRiskMetrics(raw: Record<string, unknown>): {
  annualizedReturn: number;
  annualizedVolatility: number;
  sharpe: number;
  sortino: number;
  calmar: number;
  finalEquity: number;
  maxDrawdown?: number;
  maxDrawdownDurationBars?: number;
} {
  return {
    annualizedReturn: Number(raw.annualized_return ?? 0),
    annualizedVolatility: Number(raw.annualized_volatility ?? 0),
    sharpe: Number(raw.sharpe ?? 0),
    sortino: Number(raw.sortino ?? 0),
    calmar: Number(raw.calmar ?? 0),
    finalEquity: Number(raw.final_equity ?? 1),
    maxDrawdown:
      raw.max_drawdown == null ? undefined : Number(raw.max_drawdown),
    maxDrawdownDurationBars:
      raw.max_drawdown_duration_bars == null
        ? undefined
        : Number(raw.max_drawdown_duration_bars),
  };
}

async function loadModelEvaluation(modelName: string) {
  const modelDir = path.join(EVALUATION_DIR, modelName);

  const [
    thresholdRows,
    regimeRows,
    unfilteredCurveRows,
    filteredCurveRows,
    summaryRaw,
  ] = await Promise.all([
    readCsvRows<ThresholdGridRow>(
      path.join(EVALUATION_DIR, `${modelName}_threshold_grid.csv`)
    ),
    readCsvRows<RegimeRow>(
      path.join(modelDir, "volatility_regime_metrics.csv")
    ),
    readCsvRows<CurveRow>(path.join(modelDir, "backtest_unfiltered_curve.csv")),
    readCsvRows<CurveRow>(path.join(modelDir, "backtest_filtered_curve.csv")),
    readJsonFile<Record<string, unknown>>(
      path.join(modelDir, "evaluation_summary.json")
    ),
  ]);

  const thresholdSweep = thresholdRows.map((row) => ({
    threshold: toNullableNumber(row.threshold),
    sharpe: toNumber(row.sharpe),
    sortino: toNumber(row.sortino),
    calmar: toNumber(row.calmar),
    finalEquity: toNumber(row.final_equity),
    maxDrawdown: toNumber(row.max_drawdown),
  }));

  const volatilityRegimes = regimeRows.map((row) => ({
    regime: row.regime,
    count: toNumber(row.count),
    accuracy: toNumber(row.accuracy),
    macroF1: toNumber(row.macro_f1),
  }));

  const mapCurve = (curveRows: CurveRow[]) => {
    const parsed = curveRows.map((row) => ({
      timestamp: row.timestamp,
      strategyReturn: toNumber(row.strategy_return),
      equityCurve: toNumber(row.equity_curve),
      benchmarkCurve: toNumber(row.benchmark_curve),
    }));

    const activeBars = parsed.filter(
      (row) => Math.abs(row.strategyReturn) > 1e-12
    ).length;

    return {
      curve: downsampleCurve(parsed),
      strategyActivityRatio: parsed.length > 0 ? activeBars / parsed.length : 0,
    };
  };

  const unfilteredCurve = mapCurve(unfilteredCurveRows);
  const filteredCurve = mapCurve(filteredCurveRows);

  const summary = summaryRaw ?? {};
  const degradationRaw =
    (summary.degradation as Record<string, unknown> | undefined) ?? undefined;

  const backtestUnfilteredRaw =
    (summary.backtest_unfiltered as Record<string, unknown> | undefined) ??
    undefined;
  const backtestFilteredRaw =
    (summary.backtest_filtered as Record<string, unknown> | undefined) ??
    undefined;

  return ModelEvaluationSchema.parse({
    degradation: degradationRaw
      ? {
          accuracyDegradationPct: Number(
            degradationRaw.accuracy_degradation_pct ?? 0
          ),
          macroF1DegradationPct: Number(
            degradationRaw.macro_f1_degradation_pct ?? 0
          ),
        }
      : null,
    thresholdSweep,
    volatilityRegimes,
    backtestUnfiltered: backtestUnfilteredRaw
      ? {
          strategyMetrics: normalizeRiskMetrics(
            (backtestUnfilteredRaw.strategy_metrics as Record<
              string,
              unknown
            >) ?? {}
          ),
          benchmarkMetrics: normalizeRiskMetrics(
            (backtestUnfilteredRaw.benchmark_metrics as Record<
              string,
              unknown
            >) ?? {}
          ),
          strategyActivityRatio: unfilteredCurve.strategyActivityRatio,
          curve: unfilteredCurve.curve,
        }
      : null,
    backtestFiltered: backtestFilteredRaw
      ? {
          strategyMetrics: normalizeRiskMetrics(
            (backtestFilteredRaw.strategy_metrics as Record<string, unknown>) ??
              {}
          ),
          benchmarkMetrics: normalizeRiskMetrics(
            (backtestFilteredRaw.benchmark_metrics as Record<
              string,
              unknown
            >) ?? {}
          ),
          strategyActivityRatio: filteredCurve.strategyActivityRatio,
          curve: filteredCurve.curve,
        }
      : null,
  });
}

async function loadDataQualitySummary() {
  const raw = await readJsonFile<Record<string, unknown>>(
    DATA_QUALITY_SUMMARY_PATH
  );
  if (!raw) {
    return null;
  }

  return DataQualitySummarySchema.parse({
    rows: Number(raw.rows ?? 0),
    tickers: Number(raw.tickers ?? 0),
    featureCount: Number(raw.feature_count ?? 0),
    relatedFeatureCount: Number(raw.related_feature_count ?? 0),
    timestampMin: raw.timestamp_min == null ? null : String(raw.timestamp_min),
    timestampMax: raw.timestamp_max == null ? null : String(raw.timestamp_max),
    labelDistribution: Object.fromEntries(
      Object.entries(
        (raw.label_distribution as Record<string, unknown>) ?? {}
      ).map(([key, value]) => [key, Number(value)])
    ),
  });
}

async function loadTickerModelingReport(): Promise<TickerModelingRow[]> {
  const rows = await readCsvRows<TickerModelingCsvRow>(
    TICKER_MODELING_REPORT_PATH
  );

  return rows
    .filter((row) => row.ticker && row.ticker.trim() !== "")
    .map((row) =>
      TickerModelingRowSchema.parse({
        ticker: row.ticker.trim(),
        rowsAfterCleaning: toNumber(row.rows_after_cleaning),
        rowsAfterFeatures: toNumber(row.rows_after_features),
        rowsAfterLabels: toNumber(row.rows_after_labels),
        label0Pct: toNumber(row.label_0_pct),
        label1Pct: toNumber(row.label_1_pct),
        label2Pct: toNumber(row.label_2_pct),
      })
    );
}

async function loadTickerCleaningReport(): Promise<TickerCleaningRow[]> {
  const rows = await readCsvRows<TickerCleaningCsvRow>(
    TICKER_CLEANING_REPORT_PATH
  );

  return rows
    .filter((row) => row.ticker && row.ticker.trim() !== "")
    .map((row) =>
      TickerCleaningRowSchema.parse({
        ticker: row.ticker.trim(),
        rowsBefore: toNumber(row.rows_before),
        rowsAfter: toNumber(row.rows_after),
        droppedRows: toNumber(row.dropped_rows),
        dropPct: toNumber(row.drop_pct),
        droppedDuplicateTs: toNumber(row.dropped_duplicate_ts),
        droppedNonSession: toNumber(row.dropped_non_session),
        droppedInvalidOhlcv: toNumber(row.dropped_invalid_ohlcv),
        droppedOutlierReturn: toNumber(row.dropped_outlier_return),
        droppedOutlierRange: toNumber(row.dropped_outlier_range),
        droppedLowCoverageSession: toNumber(row.dropped_low_coverage_session),
        keptSessions: toNumber(row.kept_sessions),
      })
    );
}

async function loadTrainingHistoryByModel(
  model: string
): Promise<TrainingPoint[]> {
  let files: string[] = [];

  try {
    files = await fs.readdir(TRAINING_LOGS_DIR);
  } catch {
    return [];
  }

  const modelFiles = files
    .filter(
      (fileName) =>
        fileName.startsWith(`${model}_fold_`) &&
        fileName.endsWith("_history.csv")
    )
    .map((fileName) => path.join(TRAINING_LOGS_DIR, fileName));

  if (modelFiles.length === 0) {
    return [];
  }

  const foldHistories = await Promise.all(
    modelFiles.map(async (filePath) => {
      const csvRaw = await fs.readFile(filePath, "utf-8");
      const parsed = Papa.parse<CsvRow>(csvRaw, {
        header: true,
        skipEmptyLines: true,
      });

      return parsed.data.map((row) => ({
        epoch: toNumber(row.epoch),
        trainLoss: toNumber(row.train_loss),
        valLoss: toNumber(row.val_loss),
        trainMacroF1: toNumber(row.train_macro_f1),
        valMacroF1: toNumber(row.val_macro_f1),
        learningRate: toNumber(row.learning_rate),
      }));
    })
  );

  const byEpoch = new Map<number, { count: number; value: TrainingPoint }>();

  for (const fold of foldHistories) {
    for (const point of fold) {
      const existing = byEpoch.get(point.epoch);
      if (!existing) {
        byEpoch.set(point.epoch, {
          count: 1,
          value: { ...point },
        });
        continue;
      }

      existing.count += 1;
      existing.value.trainLoss += point.trainLoss;
      existing.value.valLoss += point.valLoss;
      existing.value.trainMacroF1 += point.trainMacroF1;
      existing.value.valMacroF1 += point.valMacroF1;
      existing.value.learningRate += point.learningRate;
    }
  }

  return [...byEpoch.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([, entry]) => ({
      epoch: entry.value.epoch,
      trainLoss: entry.value.trainLoss / entry.count,
      valLoss: entry.value.valLoss / entry.count,
      trainMacroF1: entry.value.trainMacroF1 / entry.count,
      valMacroF1: entry.value.valMacroF1 / entry.count,
      learningRate: entry.value.learningRate / entry.count,
    }));
}

export async function loadDashboardPayload(): Promise<DashboardPayload> {
  const [
    resultsRaw,
    kpiReportRaw,
    overfitReportRaw,
    sharpeSummaryRaw,
    dataQualitySummary,
    tickerModelingReport,
    tickerCleaningReport,
  ] = await Promise.all([
    readJsonFile<unknown>(RESULTS_SUMMARY_PATH),
    readJsonFile<unknown>(KPI_REPORT_PATH),
    readJsonFile<unknown>(OVERFIT_REPORT_PATH),
    readJsonFile<unknown>(SHARPE_SUMMARY_PATH),
    loadDataQualitySummary(),
    loadTickerModelingReport(),
    loadTickerCleaningReport(),
  ]);

  const parsedResults = ResultsSummaryRawSchema.parse(
    resultsRaw ?? { models: {}, feature_columns: [] }
  );
  const kpiReport = kpiReportRaw
    ? KpiAccuracyReportSchema.parse(kpiReportRaw)
    : null;
  const overfitReport = overfitReportRaw
    ? OverfitHealthReportSchema.parse(overfitReportRaw)
    : null;
  const sharpeSummary = sharpeSummaryRaw
    ? SharpeSummarySchema.parse(sharpeSummaryRaw)
    : null;

  const models = Object.entries(parsedResults.models).map(
    ([modelName, modelPayload]) => {
      const tunedValMacroF1Mean = mean(
        modelPayload.folds.map(
          (fold) => fold.tuned_val_macro_f1 ?? fold.best_val_macro_f1
        )
      );
      const bestValMacroF1Mean = mean(
        modelPayload.folds.map((fold) => fold.best_val_macro_f1)
      );

      const decisionBiasesMean = [0, 0, 0];
      for (const fold of modelPayload.folds) {
        const biases = fold.decision_class_biases ?? [0, 0, 0];
        decisionBiasesMean[0] += Number(biases[0] ?? 0);
        decisionBiasesMean[1] += Number(biases[1] ?? 0);
        decisionBiasesMean[2] += Number(biases[2] ?? 0);
      }

      if (modelPayload.folds.length > 0) {
        decisionBiasesMean[0] /= modelPayload.folds.length;
        decisionBiasesMean[1] /= modelPayload.folds.length;
        decisionBiasesMean[2] /= modelPayload.folds.length;
      }

      const confusion = averageConfusionMatrix(
        modelPayload.folds.map((fold) => fold.test_confusion_matrix)
      );

      const bestEpoch = Math.max(
        ...modelPayload.folds.map((fold) => fold.best_epoch),
        0
      );

      return {
        model: modelName,
        lossMean: modelPayload.summary.test_loss.mean,
        lossStd: modelPayload.summary.test_loss.std,
        accuracyMean: modelPayload.summary.test_accuracy.mean,
        accuracyStd: modelPayload.summary.test_accuracy.std,
        macroF1Mean: modelPayload.summary.test_macro_f1.mean,
        macroF1Std: modelPayload.summary.test_macro_f1.std,
        bestValMacroF1Mean,
        tunedValMacroF1Mean,
        baselineAccuracyMean: modelPayload.summary.baseline_accuracy.mean,
        baselineMacroF1Mean: modelPayload.summary.baseline_macro_f1.mean,
        deltaVsBaseline:
          modelPayload.summary.test_accuracy.mean -
          modelPayload.summary.baseline_accuracy.mean,
        deltaF1VsBaseline:
          modelPayload.summary.test_macro_f1.mean -
          modelPayload.summary.baseline_macro_f1.mean,
        bestEpoch,
        totalFolds: modelPayload.folds.length,
        decisionBiasesMean,
        confusionMatrix: confusion,
      };
    }
  );

  const trainingHistoryEntries = await Promise.all(
    models.map(async (model) => {
      const history = await loadTrainingHistoryByModel(model.model);
      return [model.model, history] as const;
    })
  );

  const modelEvaluationEntries = await Promise.all(
    models.map(async (model) => {
      const evaluation = await loadModelEvaluation(model.model);
      return [model.model, evaluation] as const;
    })
  );

  const mcnemar = Object.fromEntries(
    Object.entries(parsedResults.mcnemar ?? {}).map(([pairName, rows]) => [
      pairName,
      rows.map((row) => ({
        n01: row.n01,
        n10: row.n10,
        chi2: row.chi2,
        pValue: row.p_value,
        foldId: row.fold_id,
      })),
    ])
  );

  const payload = DashboardPayloadSchema.parse({
    generatedAt: new Date().toISOString(),
    featureColumns: parsedResults.feature_columns,
    models,
    dataQualitySummary,
    tickerModelingReport,
    tickerCleaningReport,
    kpiReport,
    overfitReport,
    sharpeSummary,
    mcnemar,
    modelEvaluations: Object.fromEntries(modelEvaluationEntries),
    trainingHistory: Object.fromEntries(trainingHistoryEntries),
  });

  return payload;
}
