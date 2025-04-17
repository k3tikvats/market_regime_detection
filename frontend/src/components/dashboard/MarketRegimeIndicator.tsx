
import { cn } from "@/lib/utils";
import { TrendingUp, AlertTriangle, BarChart } from "lucide-react";

interface MarketRegimeIndicatorProps {
  regime: string;
  status: "favorable" | "neutral" | "risk";
  className?: string;
  style?: React.CSSProperties;
}

export function MarketRegimeIndicator({ 
  regime, 
  status, 
  className,
  style
}: MarketRegimeIndicatorProps) {
  return (
    <div className={cn("rounded-lg border p-6 animate-scale-in", className)} style={style}>
      <div className="space-y-4">
        <h3 className="text-sm font-medium text-muted-foreground">Current Market Regime</h3>
        <div className="flex items-center gap-4">
          <StatusIndicator status={status} />
          <div>
            <h2 className="text-2xl font-bold tracking-tight">{regime}</h2>
            <p className="text-sm text-muted-foreground">
              {status === "favorable" && "Optimal conditions for trading"}
              {status === "neutral" && "Exercise caution with positions"}
              {status === "risk" && "High risk environment - reduce exposure"}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

function StatusIndicator({ status }: { status: "favorable" | "neutral" | "risk" }) {
  return (
    <div
      className={cn(
        "h-14 w-14 rounded-full flex items-center justify-center",
        status === "favorable" && "bg-regime-favorable/20 text-regime-favorable",
        status === "neutral" && "bg-regime-neutral/20 text-regime-neutral",
        status === "risk" && "bg-regime-risk/20 text-regime-risk"
      )}
    >
      {status === "favorable" && <TrendingUp size={24} />}
      {status === "neutral" && <BarChart size={24} />}
      {status === "risk" && <AlertTriangle size={24} />}
    </div>
  );
}
