"use client";

import { ArrowLeftIcon } from "@radix-ui/react-icons";
import { Button, Flex } from "@radix-ui/themes";
import Link from "next/link";

import { GuidePanel } from "@/components/dashboard/guide-panel";
import { ThemeToggle } from "@/components/theme-toggle";
import { useDashboardData } from "@/hooks/use-dashboard-data";

export function HelpPage() {
  const { data } = useDashboardData();

  const neutralClassPct =
    data?.dataQualitySummary?.labelDistribution?.["1"] ?? null;

  return (
    <Flex direction="column" gap="5">
      <Flex justify="between" align="center" wrap="wrap" gap="3">
        <Link href="/" prefetch>
          <Button variant="soft" color="teal" size="3">
            <ArrowLeftIcon />
            Back to dashboard
          </Button>
        </Link>
        <ThemeToggle />
      </Flex>
      <GuidePanel neutralClassPct={neutralClassPct} />
    </Flex>
  );
}
