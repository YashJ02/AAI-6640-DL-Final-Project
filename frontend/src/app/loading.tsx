import { Container } from "@radix-ui/themes";

import { LoadingState } from "@/components/ui/loading-state";

export default function Loading() {
  return (
    <main className="dashboard-shell">
      <Container size="4" py="6">
        <LoadingState />
      </Container>
    </main>
  );
}
