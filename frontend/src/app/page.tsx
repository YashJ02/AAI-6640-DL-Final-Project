import { Container } from "@radix-ui/themes";

import { DashboardPage } from "@/components/dashboard/dashboard-page";

export default function Home() {
  return (
    <main className="dashboard-shell">
      <Container size="4" py="6">
        <DashboardPage />
      </Container>
    </main>
  );
}
