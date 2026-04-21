"use client";

import { motion } from "motion/react";
import { Card, Flex, Heading, Text } from "@radix-ui/themes";
import { Fragment } from "react";

type ConfusionMatrixProps = {
  matrix: number[][];
  modelName: string;
};

const rowLabels = ["True Down", "True Neutral", "True Up"];
const colLabels = ["Pred Down", "Pred Neutral", "Pred Up"];

export function ConfusionMatrix({ matrix, modelName }: ConfusionMatrixProps) {
  const max = Math.max(...matrix.flat(), 1);

  return (
    <Card>
      <Flex direction="column" gap="3">
        <Heading size="4">Confusion Matrix - {modelName}</Heading>
        <div className="matrix-wrap">
          <div className="matrix-grid">
            <div />
            {colLabels.map((label) => (
              <Text
                key={label}
                weight="medium"
                size="2"
                className="matrix-label-top"
              >
                {label}
              </Text>
            ))}

            {matrix.map((row, rowIndex) => (
              <Fragment key={`row-${rowLabels[rowIndex]}`}>
                <Text
                  key={`row-label-${rowLabels[rowIndex]}`}
                  weight="medium"
                  size="2"
                  className="matrix-label-left"
                >
                  {rowLabels[rowIndex]}
                </Text>
                {row.map((value, colIndex) => {
                  const intensity = value / max;
                  return (
                    <motion.div
                      key={`cell-${rowIndex}-${colIndex}`}
                      className="matrix-cell"
                      initial={{ opacity: 0, scale: 0.9 }}
                      animate={{ opacity: 1, scale: 1 }}
                      transition={{
                        duration: 0.35,
                        delay: (rowIndex * row.length + colIndex) * 0.03,
                      }}
                      style={{
                        background: `rgba(13, 148, 136, ${
                          0.18 + intensity * 0.72
                        })`,
                      }}
                    >
                      <Text size="3" weight="bold">
                        {value.toFixed(0)}
                      </Text>
                    </motion.div>
                  );
                })}
              </Fragment>
            ))}
          </div>
        </div>
      </Flex>
    </Card>
  );
}
