"use client";

import { Badge, Card, Flex, Heading, ScrollArea, Text } from "@radix-ui/themes";

type FeaturePillListProps = {
  features: string[];
  query: string;
};

export function FeaturePillList({ features, query }: FeaturePillListProps) {
  const filtered = features.filter((feature) =>
    feature.toLowerCase().includes(query.trim().toLowerCase())
  );

  return (
    <Card>
      <Flex direction="column" gap="3">
        <Heading size="4">Feature Space</Heading>
        <Text size="2" color="gray">
          {filtered.length} / {features.length} features shown
        </Text>
        <ScrollArea
          type="always"
          scrollbars="vertical"
          style={{ maxHeight: 170 }}
        >
          <Flex wrap="wrap" gap="2">
            {filtered.map((feature) => (
              <Badge key={feature} color="teal" variant="soft" radius="full">
                {feature}
              </Badge>
            ))}
          </Flex>
        </ScrollArea>
      </Flex>
    </Card>
  );
}
