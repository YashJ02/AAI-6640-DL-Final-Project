"use client";

import { zodResolver } from "@hookform/resolvers/zod";
import { Button, Card, Flex, Select, Text, TextField } from "@radix-ui/themes";
import { useEffect } from "react";
import { Controller, useForm } from "react-hook-form";
import { z } from "zod";

import { useDashboardStore, type MetricMode } from "@/store/dashboard-store";

const FilterSchema = z.object({
  model: z.string().min(1, "Model is required"),
  metricMode: z.enum(["macro_f1", "accuracy"]),
  featureSearch: z.string().max(60, "Search text is too long").optional(),
});

type FilterValues = z.infer<typeof FilterSchema>;

type ControlPanelProps = {
  models: string[];
};

export function ControlPanel({ models }: ControlPanelProps) {
  const {
    selectedModel,
    metricMode,
    featureSearch,
    setSelectedModel,
    setMetricMode,
    setFeatureSearch,
  } = useDashboardStore();

  const fallbackModel = selectedModel ?? models[0] ?? "";

  const { control, register, handleSubmit, reset, formState } =
    useForm<FilterValues>({
      resolver: zodResolver(FilterSchema),
      defaultValues: {
        model: fallbackModel,
        metricMode,
        featureSearch,
      },
    });

  useEffect(() => {
    reset({
      model: fallbackModel,
      metricMode,
      featureSearch,
    });
  }, [fallbackModel, metricMode, featureSearch, reset]);

  const onSubmit = (values: FilterValues) => {
    setSelectedModel(values.model);
    setMetricMode(values.metricMode as MetricMode);
    setFeatureSearch(values.featureSearch ?? "");
  };

  return (
    <Card>
      <form onSubmit={handleSubmit(onSubmit)}>
        <Flex direction="column" gap="4">
          <Text weight="bold" size="3">
            Dashboard Controls
          </Text>

          <Flex direction={{ initial: "column", md: "row" }} gap="3">
            <Controller
              control={control}
              name="model"
              render={({ field }) => (
                <Select.Root value={field.value} onValueChange={field.onChange}>
                  <Select.Trigger radius="large" placeholder="Select model" />
                  <Select.Content>
                    {models.map((model) => (
                      <Select.Item key={model} value={model}>
                        {model}
                      </Select.Item>
                    ))}
                  </Select.Content>
                </Select.Root>
              )}
            />

            <Controller
              control={control}
              name="metricMode"
              render={({ field }) => (
                <Select.Root value={field.value} onValueChange={field.onChange}>
                  <Select.Trigger radius="large" placeholder="Metric mode" />
                  <Select.Content>
                    <Select.Item value="macro_f1">Macro F1 Focus</Select.Item>
                    <Select.Item value="accuracy">Accuracy Focus</Select.Item>
                  </Select.Content>
                </Select.Root>
              )}
            />

            <TextField.Root
              radius="large"
              placeholder="Search features"
              {...register("featureSearch")}
            />
          </Flex>

          {formState.errors.featureSearch ? (
            <Text size="1" color="red">
              {formState.errors.featureSearch.message}
            </Text>
          ) : null}

          <Flex gap="2" justify="end">
            <Button
              type="button"
              variant="soft"
              color="gray"
              onClick={() => reset()}
            >
              Reset
            </Button>
            <Button type="submit" color="teal">
              Apply Filters
            </Button>
          </Flex>
        </Flex>
      </form>
    </Card>
  );
}
