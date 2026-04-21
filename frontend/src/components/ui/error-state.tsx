"use client";

import { ExclamationTriangleIcon, ReloadIcon } from "@radix-ui/react-icons";
import { Button, Callout, Flex } from "@radix-ui/themes";

type ErrorStateProps = {
  message: string;
  onRetry: () => void;
};

export function ErrorState({ message, onRetry }: ErrorStateProps) {
  return (
    <Flex direction="column" gap="4">
      <Callout.Root color="red" role="alert">
        <Callout.Icon>
          <ExclamationTriangleIcon />
        </Callout.Icon>
        <Callout.Text>{message}</Callout.Text>
      </Callout.Root>
      <Button color="teal" variant="solid" onClick={onRetry}>
        <ReloadIcon />
        Retry
      </Button>
    </Flex>
  );
}
