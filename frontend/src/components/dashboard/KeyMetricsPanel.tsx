
import { cn } from "@/lib/utils";
import { ArrowUp, ArrowDown, ArrowRight } from "lucide-react";

interface MetricProps {
  title: string;
  value: string | number;
  indicator?: "up" | "down" | "neutral";
  description?: string;
}

interface KeyMetricsPanelProps {
  className?: string;
  style?: React.CSSProperties;
}

export function KeyMetricsPanel({ className, style }: KeyMetricsPanelProps) {
  return (
    <div className={cn("rounded-lg border p-6", className)} style={style}>
      <div className="mb-4">
        <h3 className="text-sm font-medium text-muted-foreground">Key Market Metrics</h3>
      </div>
      <div className="grid gap-6 md:grid-cols-3">
        <Metric
          title="Volatility"
          value="32.4"
          indicator="up"
          description="Increasing by 4.2%"
        />
        <Metric
          title="Liquidity"
          value="67.8"
          indicator="down"
          description="Decreasing by 2.6%"
        />
        <Metric
          title="Directional Bias"
          value="Bearish"
          indicator="down"
          description="Strengthening bias"
        />
      </div>
    </div>
  );
}

function Metric({ title, value, indicator, description }: MetricProps) {
  return (
    <div className="space-y-2">
      <p className="text-xs font-medium text-muted-foreground">{title}</p>
      <div className="flex items-baseline gap-2">
        <p className="text-2xl font-bold">{value}</p>
        {indicator && (
          <span
            className={cn(
              "text-xs font-medium",
              indicator === "up" && "text-regime-risk",
              indicator === "down" && "text-regime-favorable",
              indicator === "neutral" && "text-regime-neutral"
            )}
          >
            {indicator === "up" && <ArrowUp size={14} />}
            {indicator === "down" && <ArrowDown size={14} />}
            {indicator === "neutral" && <ArrowRight size={14} />}
          </span>
        )}
      </div>
      {description && <p className="text-xs text-muted-foreground">{description}</p>}
    </div>
  );
}
