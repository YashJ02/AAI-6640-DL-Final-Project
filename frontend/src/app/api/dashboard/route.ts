import { NextResponse } from "next/server";

import { loadDashboardPayload } from "@/lib/server/artifacts";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const payload = await loadDashboardPayload();
    return NextResponse.json(payload);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    return NextResponse.json(
      {
        message: "Failed to load dashboard data",
        details: message,
      },
      { status: 500 }
    );
  }
}
