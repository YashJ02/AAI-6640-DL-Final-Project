"use client";

import { Button, Container, Flex, Heading, Text } from "@radix-ui/themes";

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return (
    <main className="dashboard-shell">
      <Container size="3" py="8">
        <Flex direction="column" gap="3" align="start">
          <Heading size="7">Something went wrong</Heading>
          <Text color="gray">{error.message}</Text>
          <Button color="teal" onClick={() => reset()}>
            Try again
          </Button>
        </Flex>
      </Container>
    </main>
  );
}
