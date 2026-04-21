"use client";

import { useEffect, useRef, useState, type ReactNode } from "react";

type ChartSurfaceProps = {
  height: number;
  minWidth?: number;
  children: (size: { width: number; height: number }) => ReactNode;
};

export function ChartSurface({
  height,
  minWidth = 280,
  children,
}: ChartSurfaceProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [width, setWidth] = useState(0);

  useEffect(() => {
    const node = containerRef.current;
    if (!node) {
      return;
    }

    const observer = new ResizeObserver((entries) => {
      const nextWidth = entries[0]?.contentRect.width ?? 0;
      setWidth(nextWidth > 0 ? nextWidth : 0);
    });

    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  return (
    <div ref={containerRef} style={{ width: "100%", minWidth: 0, height }}>
      {width >= minWidth ? children({ width, height }) : null}
    </div>
  );
}
