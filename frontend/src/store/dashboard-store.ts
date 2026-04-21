import { create } from "zustand";

export type MetricMode = "macro_f1" | "accuracy";
export type BacktestMode = "unfiltered" | "filtered";
export type TrainingMetric = "f1" | "loss" | "learningRate";

type DashboardState = {
  selectedModel: string | null;
  metricMode: MetricMode;
  featureSearch: string;
  backtestMode: BacktestMode;
  trainingMetric: TrainingMetric;
  setSelectedModel: (model: string) => void;
  setMetricMode: (metricMode: MetricMode) => void;
  setFeatureSearch: (query: string) => void;
  setBacktestMode: (mode: BacktestMode) => void;
  setTrainingMetric: (metric: TrainingMetric) => void;
};

export const useDashboardStore = create<DashboardState>((set) => ({
  selectedModel: null,
  metricMode: "macro_f1",
  featureSearch: "",
  backtestMode: "unfiltered",
  trainingMetric: "f1",
  setSelectedModel: (selectedModel) => set({ selectedModel }),
  setMetricMode: (metricMode) => set({ metricMode }),
  setFeatureSearch: (featureSearch) => set({ featureSearch }),
  setBacktestMode: (backtestMode) => set({ backtestMode }),
  setTrainingMetric: (trainingMetric) => set({ trainingMetric }),
}));
