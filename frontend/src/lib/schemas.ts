import { z } from "zod";

const MetricSchema = z.object({
  mean: z.number(),
  std: z.number(),
});

const FoldSchema = z.object({
  model_name: z.string(),
  fold_id: z.number(),
  best_epoch: z.number(),
  best_val_macro_f1: z.number(),
  best_checkpoint_val_macro_f1: z.number().optional(),
  tuned_val_macro_f1: z.number().optional(),
  decision_class_biases: z.array(z.number()).optional(),
  test_loss: z.number(),
  test_accuracy: z.number(),
  test_macro_f1: z.number(),
  test_confusion_matrix: z.array(z.array(z.number())),
  history_log_path: z.string().optional(),
  checkpoint_path: z.string().optional(),
});

const McnemarRawEntrySchema = z.object({
  n01: z.number(),
  n10: z.number(),
  chi2: z.number(),
  p_value: z.number(),
  fold_id: z.number(),
});

const ModelRawSchema = z.object({
  summary: z.object({
    test_loss: MetricSchema,
    test_accuracy: MetricSchema,
    test_macro_f1: MetricSchema,
    baseline_accuracy: MetricSchema,
    baseline_macro_f1: MetricSchema,
  }),
  folds: z.array(FoldSchema),
});

export const ResultsSummaryRawSchema = z.object({
  feature_columns: z.array(z.string()),
  models: z.record(z.string(), ModelRawSchema),
  mcnemar: z.record(z.string(), z.array(McnemarRawEntrySchema)).optional(),
});

export const KpiAccuracyReportSchema = z.object({
  enabled: z.boolean(),
  enforce: z.boolean(),
  target_accuracy_min: z.number(),
  target_delta_vs_baseline_min: z.number(),
  best_passing_model: z.string().nullable().optional(),
  any_model_passed: z.boolean().optional(),
  best_by_accuracy: z
    .object({
      model: z.string(),
      accuracy: z.number(),
    })
    .optional(),
  best_by_delta_vs_baseline: z
    .object({
      model: z.string(),
      delta_vs_baseline: z.number(),
    })
    .optional(),
  models: z.record(
    z.string(),
    z.object({
      test_accuracy: z.number(),
      baseline_accuracy: z.number(),
      delta_vs_baseline: z.number(),
      pass_accuracy_target: z.boolean(),
      pass_baseline_gap: z.boolean(),
      pass_all: z.boolean(),
    })
  ),
});

export const OverfitModelSchema = z.object({
  epochs_ran: z.number(),
  best_epoch_by_val_f1: z.number(),
  best_val_macro_f1: z.number(),
  best_train_macro_f1_same_epoch: z.number(),
  best_epoch_gap_val_minus_train: z.number(),
  last_epoch_train_macro_f1: z.number(),
  last_epoch_val_macro_f1: z.number(),
  last_epoch_gap_val_minus_train: z.number(),
});

export const OverfitHealthReportSchema = z.record(
  z.string(),
  OverfitModelSchema
);

export const SharpeEntrySchema = z.object({
  best_threshold: z.number().nullable(),
  best_sharpe: z.number(),
  best_sortino: z.number(),
  best_calmar: z.number(),
  best_final_equity: z.number(),
  best_max_drawdown: z.number(),
});

export const SharpeSummarySchema = z.record(z.string(), SharpeEntrySchema);

export const TrainingPointSchema = z.object({
  epoch: z.number(),
  trainLoss: z.number(),
  valLoss: z.number(),
  trainMacroF1: z.number(),
  valMacroF1: z.number(),
  learningRate: z.number(),
});

export const DashboardModelSchema = z.object({
  model: z.string(),
  accuracyMean: z.number(),
  accuracyStd: z.number(),
  lossMean: z.number(),
  lossStd: z.number(),
  macroF1Mean: z.number(),
  macroF1Std: z.number(),
  bestValMacroF1Mean: z.number(),
  tunedValMacroF1Mean: z.number(),
  baselineAccuracyMean: z.number(),
  baselineMacroF1Mean: z.number(),
  deltaVsBaseline: z.number(),
  deltaF1VsBaseline: z.number(),
  bestEpoch: z.number(),
  totalFolds: z.number(),
  decisionBiasesMean: z.array(z.number()).length(3),
  confusionMatrix: z.array(z.array(z.number())),
});

export const DataQualitySummarySchema = z.object({
  rows: z.number(),
  tickers: z.number(),
  featureCount: z.number(),
  relatedFeatureCount: z.number(),
  timestampMin: z.string().nullable(),
  timestampMax: z.string().nullable(),
  labelDistribution: z.record(z.string(), z.number()),
});

export const TickerModelingRowSchema = z.object({
  ticker: z.string(),
  rowsAfterCleaning: z.number(),
  rowsAfterFeatures: z.number(),
  rowsAfterLabels: z.number(),
  label0Pct: z.number(),
  label1Pct: z.number(),
  label2Pct: z.number(),
});

export const TickerCleaningRowSchema = z.object({
  ticker: z.string(),
  rowsBefore: z.number(),
  rowsAfter: z.number(),
  droppedRows: z.number(),
  dropPct: z.number(),
  droppedDuplicateTs: z.number(),
  droppedNonSession: z.number(),
  droppedInvalidOhlcv: z.number(),
  droppedOutlierReturn: z.number(),
  droppedOutlierRange: z.number(),
  droppedLowCoverageSession: z.number(),
  keptSessions: z.number(),
});

export const McnemarEntrySchema = z.object({
  n01: z.number(),
  n10: z.number(),
  chi2: z.number(),
  pValue: z.number(),
  foldId: z.number(),
});

export const ThresholdSweepPointSchema = z.object({
  threshold: z.number().nullable(),
  sharpe: z.number(),
  sortino: z.number(),
  calmar: z.number(),
  finalEquity: z.number(),
  maxDrawdown: z.number(),
});

export const VolatilityRegimeMetricSchema = z.object({
  regime: z.string(),
  count: z.number(),
  accuracy: z.number(),
  macroF1: z.number(),
});

export const CurvePointSchema = z.object({
  timestamp: z.string(),
  strategyReturn: z.number(),
  equityCurve: z.number(),
  benchmarkCurve: z.number(),
});

export const RiskMetricSummarySchema = z.object({
  annualizedReturn: z.number(),
  annualizedVolatility: z.number(),
  sharpe: z.number(),
  sortino: z.number(),
  calmar: z.number(),
  finalEquity: z.number(),
  maxDrawdown: z.number().optional(),
  maxDrawdownDurationBars: z.number().optional(),
});

export const BacktestViewSchema = z.object({
  strategyMetrics: RiskMetricSummarySchema,
  benchmarkMetrics: RiskMetricSummarySchema,
  strategyActivityRatio: z.number(),
  curve: z.array(CurvePointSchema),
});

export const ModelEvaluationSchema = z.object({
  degradation: z
    .object({
      accuracyDegradationPct: z.number(),
      macroF1DegradationPct: z.number(),
    })
    .nullable(),
  thresholdSweep: z.array(ThresholdSweepPointSchema),
  volatilityRegimes: z.array(VolatilityRegimeMetricSchema),
  backtestUnfiltered: BacktestViewSchema.nullable(),
  backtestFiltered: BacktestViewSchema.nullable(),
});

export const DashboardPayloadSchema = z.object({
  generatedAt: z.string(),
  featureColumns: z.array(z.string()),
  models: z.array(DashboardModelSchema),
  dataQualitySummary: DataQualitySummarySchema.nullable(),
  tickerModelingReport: z.array(TickerModelingRowSchema),
  tickerCleaningReport: z.array(TickerCleaningRowSchema),
  kpiReport: KpiAccuracyReportSchema.nullable(),
  overfitReport: OverfitHealthReportSchema.nullable(),
  sharpeSummary: SharpeSummarySchema.nullable(),
  mcnemar: z.record(z.string(), z.array(McnemarEntrySchema)),
  modelEvaluations: z.record(z.string(), ModelEvaluationSchema),
  trainingHistory: z.record(z.string(), z.array(TrainingPointSchema)),
});

export type DashboardPayload = z.infer<typeof DashboardPayloadSchema>;
export type DashboardModel = z.infer<typeof DashboardModelSchema>;
export type TrainingPoint = z.infer<typeof TrainingPointSchema>;
export type ModelEvaluation = z.infer<typeof ModelEvaluationSchema>;
export type McnemarEntry = z.infer<typeof McnemarEntrySchema>;
export type BacktestView = z.infer<typeof BacktestViewSchema>;
export type KpiReport = z.infer<typeof KpiAccuracyReportSchema>;
export type OverfitReport = z.infer<typeof OverfitHealthReportSchema>;
export type SharpeSummary = z.infer<typeof SharpeSummarySchema>;
export type TickerModelingRow = z.infer<typeof TickerModelingRowSchema>;
export type TickerCleaningRow = z.infer<typeof TickerCleaningRowSchema>;
export type VolatilityRegimeMetric = z.infer<typeof VolatilityRegimeMetricSchema>;
export type ThresholdSweepPoint = z.infer<typeof ThresholdSweepPointSchema>;
export type CurvePoint = z.infer<typeof CurvePointSchema>;
export type RiskMetricSummary = z.infer<typeof RiskMetricSummarySchema>;
export type DataQualitySummary = z.infer<typeof DataQualitySummarySchema>;
