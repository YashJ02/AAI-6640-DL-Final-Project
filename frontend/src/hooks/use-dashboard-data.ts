"use client";

import { useQuery } from "@tanstack/react-query";

import { httpClient } from "@/lib/http";
import { DashboardPayloadSchema, type DashboardPayload } from "@/lib/schemas";

async function fetchDashboardData(): Promise<DashboardPayload> {
  const response = await httpClient.get("/dashboard");
  return DashboardPayloadSchema.parse(response.data);
}

export function useDashboardData() {
  return useQuery({
    queryKey: ["dashboard-data"],
    queryFn: fetchDashboardData,
    staleTime: 60_000,
    gcTime: 300_000,
    retry: 2,
    retryDelay: (attempt) => Math.min(1000 * 2 ** attempt, 8000),
  });
}
