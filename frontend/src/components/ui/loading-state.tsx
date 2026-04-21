"use client";

import { Card, Flex, Skeleton } from "@radix-ui/themes";

export function LoadingState() {
  return (
    <Flex direction="column" gap="4">
      <Card>
        <Skeleton loading width="100%" height="28px" />
      </Card>
      <Card>
        <Skeleton loading width="100%" height="260px" />
      </Card>
      <Card>
        <Skeleton loading width="100%" height="260px" />
      </Card>
    </Flex>
  );
}
