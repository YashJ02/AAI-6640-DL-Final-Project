"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Theme } from "@radix-ui/themes";
import { ThemeProvider } from "next-themes";
import { useState, type ReactNode } from "react";

function RadixThemeBridge({ children }: { children: ReactNode }) {
  return (
    <Theme
      accentColor="teal"
      grayColor="sand"
      radius="large"
      scaling="100%"
      appearance="inherit"
    >
      {children}
    </Theme>
  );
}

export function AppProviders({ children }: { children: ReactNode }) {
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            refetchOnWindowFocus: false,
            refetchOnReconnect: true,
          },
        },
      })
  );

  return (
    <ThemeProvider attribute="class" defaultTheme="system" enableSystem>
      <RadixThemeBridge>
        <QueryClientProvider client={queryClient}>
          {children}
        </QueryClientProvider>
      </RadixThemeBridge>
    </ThemeProvider>
  );
}
