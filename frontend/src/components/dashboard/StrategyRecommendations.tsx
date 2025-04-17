
import { cn } from "@/lib/utils";
import { useState } from "react";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Lightbulb, ShieldAlert, TrendingUp } from "lucide-react";

interface StrategyProps {
  title: string;
  description: string;
  expectedReturn: string;
  risk: "low" | "medium" | "high";
}

const beginnerStrategies: StrategyProps[] = [
  {
    title: "Index ETF Accumulation",
    description: "Gradually build positions in broad market ETFs during this regime",
    expectedReturn: "6-8% annually",
    risk: "low",
  },
  {
    title: "Market Neutral Strategy",
    description: "Balanced long/short positions to reduce market exposure",
    expectedReturn: "3-5% annually",
    risk: "low",
  },
];

const advancedStrategies: StrategyProps[] = [
  {
    title: "Volatility Arbitrage",
    description: "Exploit volatility differentials between related assets",
    expectedReturn: "15-20% annually",
    risk: "high",
  },
  {
    title: "Momentum Factor Tilt",
    description: "Overweight momentum stocks with protective options collar",
    expectedReturn: "10-12% annually",
    risk: "medium",
  },
  {
    title: "Mean-Reversion Trading",
    description: "Target oversold conditions in quality names with defined risk",
    expectedReturn: "8-14% annually",
    risk: "medium",
  },
];

interface StrategyRecommendationsProps {
  className?: string;
  style?: React.CSSProperties;
}

export function StrategyRecommendations({ className, style }: StrategyRecommendationsProps) {
  const [advanced, setAdvanced] = useState(false);
  const strategies = advanced ? advancedStrategies : beginnerStrategies;

  return (
    <div className={cn("rounded-lg border p-6", className)} style={style}>
      <div className="mb-4 flex items-center justify-between">
        <h3 className="text-sm font-medium text-muted-foreground">
          Strategy Recommendations
        </h3>
        <div className="flex items-center gap-2">
          <Label htmlFor="advanced-mode" className="text-xs">
            Advanced Mode
          </Label>
          <Switch
            id="advanced-mode"
            checked={advanced}
            onCheckedChange={setAdvanced}
          />
        </div>
      </div>
      <div className="space-y-4">
        {strategies.map((strategy) => (
          <Strategy key={strategy.title} {...strategy} />
        ))}
      </div>
      <div className="mt-6">
        <h4 className="mb-2 text-sm font-medium">Risk Exposure Meter</h4>
        <div className="h-3 w-full rounded-full bg-secondary overflow-hidden">
          <div 
            className="h-full w-1/4 bg-regime-favorable rounded-full"
            style={{ width: "25%" }} 
          />
        </div>
        <div className="mt-1 flex justify-between text-xs text-muted-foreground">
          <span>Conservative</span>
          <span>Balanced</span>
          <span>Aggressive</span>
        </div>
        <p className="mt-2 text-xs text-muted-foreground">
          Recommended risk exposure: <span className="font-medium">25%</span> of max position size
        </p>
      </div>
    </div>
  );
}

function Strategy({ title, description, expectedReturn, risk }: StrategyProps) {
  return (
    <div className="rounded-md bg-secondary/50 p-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Lightbulb size={16} className="text-primary" />
          <h4 className="font-medium">{title}</h4>
        </div>
        <RiskBadge risk={risk} />
      </div>
      <p className="mt-1 text-sm text-muted-foreground">{description}</p>
      <p className="mt-2 text-xs">
        Expected return: <span className="font-medium">{expectedReturn}</span>
      </p>
    </div>
  );
}

function RiskBadge({ risk }: { risk: "low" | "medium" | "high" }) {
  return (
    <span
      className={cn(
        "inline-flex rounded-full px-2 py-0.5 text-xs font-medium",
        risk === "low" && "bg-regime-favorable/20 text-regime-favorable",
        risk === "medium" && "bg-regime-neutral/20 text-regime-neutral",
        risk === "high" && "bg-regime-risk/20 text-regime-risk"
      )}
    >
      {risk}
    </span>
  );
}
