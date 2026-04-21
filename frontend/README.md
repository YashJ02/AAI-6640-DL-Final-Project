# Frontend Dashboard (Next.js + Bun)

This frontend replaces the previous Streamlit UI with a modern Next.js application.

## Stack

- Next.js 16 + React 19 + TypeScript
- Bun runtime/package manager
- Radix Themes for UI components
- React Hook Form + Zod for validation
- Recharts for animated charts
- Motion for interface animations
- TanStack Query for server-state caching
- Zustand for client-state management
- xior for HTTP client requests
- next-themes for light/dark mode

## Run

```bash
bun install
bun run dev
```

Open http://localhost:3000.

## Build

```bash
bun run build
```

## Data Source

The dashboard reads model artifacts from the parent project directory (`../artifacts`) via `src/app/api/dashboard/route.ts`.
