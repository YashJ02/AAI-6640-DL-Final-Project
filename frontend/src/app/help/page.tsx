import { Container } from "@radix-ui/themes";
import type { Metadata } from "next";

import { HelpPage } from "@/components/help/help-page";

export const metadata: Metadata = {
  title: "Help & Guide · Intraday Deep Learning Dashboard",
  description:
    "Guided tour of the AAI 6640 intraday prediction project — problem, data, methodology, research questions, and how to interpret each section of the dashboard.",
};

export default function HelpRoute() {
  return (
    <main className="dashboard-shell">
      <Container size="4" py="6">
        <HelpPage />
      </Container>
    </main>
  );
}
