
import { useState } from "react";
import { cn } from "@/lib/utils";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { Calendar } from "lucide-react";

// Sample historical data with market regimes
const historicalData = {
  "1M": [
    { date: "Mar 17", value: 100, regime: "Trending & Liquid" },
    { date: "Mar 24", value: 103, regime: "Trending & Liquid" },
    { date: "Mar 31", value: 106, regime: "Trending & Liquid" },
    { date: "Apr 07", value: 102, regime: "Choppy & Liquid", regimeChange: true },
    { date: "Apr 14", value: 99, regime: "Choppy & Liquid" },
    { date: "Apr 21", value: 95, regime: "Volatile & Illiquid", regimeChange: true },
  ],
  "3M": [
    { date: "Jan 21", value: 90, regime: "Trending & Illiquid" },
    { date: "Feb 04", value: 94, regime: "Trending & Illiquid" },
    { date: "Feb 18", value: 97, regime: "Trending & Liquid", regimeChange: true },
    { date: "Mar 04", value: 101, regime: "Trending & Liquid" },
    { date: "Mar 18", value: 105, regime: "Trending & Liquid" },
    { date: "Apr 01", value: 104, regime: "Choppy & Liquid", regimeChange: true },
    { date: "Apr 15", value: 98, regime: "Volatile & Illiquid", regimeChange: true },
  ],
  "6M": [
    { date: "Oct 21", value: 80, regime: "Volatile & Illiquid" },
    { date: "Nov 21", value: 83, regime: "Volatile & Illiquid" },
    { date: "Dec 21", value: 88, regime: "Trending & Illiquid", regimeChange: true },
    { date: "Jan 21", value: 91, regime: "Trending & Illiquid" },
    { date: "Feb 21", value: 98, regime: "Trending & Liquid", regimeChange: true },
    { date: "Mar 21", value: 104, regime: "Trending & Liquid" },
    { date: "Apr 21", value: 95, regime: "Volatile & Illiquid", regimeChange: true },
  ],
  "1Y": [
    { date: "Apr 22", value: 75, regime: "Choppy & Liquid" },
    { date: "Jun 22", value: 70, regime: "Volatile & Illiquid", regimeChange: true },
    { date: "Aug 22", value: 78, regime: "Volatile & Illiquid" },
    { date: "Oct 22", value: 73, regime: "Volatile & Illiquid" },
    { date: "Dec 22", value: 80, regime: "Trending & Illiquid", regimeChange: true },
    { date: "Feb 23", value: 95, regime: "Trending & Liquid", regimeChange: true },
    { date: "Apr 23", value: 92, regime: "Volatile & Illiquid", regimeChange: true },
  ],
};

interface HistoricalAnalysisProps {
  className?: string;
  style?: React.CSSProperties;
}

export function HistoricalAnalysis({ className, style }: HistoricalAnalysisProps) {
  const [timeframe, setTimeframe] = useState<"1M" | "3M" | "6M" | "1Y">("1M");
  const data = historicalData[timeframe];

  return (
    <div className={cn("rounded-lg border p-6", className)} style={style}>
      <div className="mb-4 flex items-center justify-between">
        <h3 className="text-sm font-medium text-muted-foreground">
          Historical Analysis
        </h3>
        <div className="flex items-center gap-2">
          <Label htmlFor="timeframe" className="text-xs flex items-center gap-1">
            <Calendar size={14} />
            Timeframe:
          </Label>
          <Select
            value={timeframe}
            onValueChange={(value) => setTimeframe(value as any)}
          >
            <SelectTrigger id="timeframe" className="h-8 w-28">
              <SelectValue placeholder="Select timeframe" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="1M">1 Month</SelectItem>
              <SelectItem value="3M">3 Months</SelectItem>
              <SelectItem value="6M">6 Months</SelectItem>
              <SelectItem value="1Y">1 Year</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>
      <div className="h-[250px]">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart
            data={data}
            margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
          >
            <defs>
              <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                <stop
                  offset="5%"
                  stopColor="hsl(var(--primary))"
                  stopOpacity={0.3}
                />
                <stop
                  offset="95%"
                  stopColor="hsl(var(--primary))"
                  stopOpacity={0}
                />
              </linearGradient>
            </defs>
            <XAxis
              dataKey="date"
              tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
              axisLine={{ stroke: "hsl(var(--border))" }}
              tickLine={false}
            />
            <YAxis
              domain={["auto", "auto"]}
              tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
              axisLine={{ stroke: "hsl(var(--border))" }}
              tickLine={false}
            />
            <CartesianGrid
              strokeDasharray="3 3"
              stroke="hsl(var(--border))"
              opacity={0.3}
              vertical={false}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "hsl(var(--card))",
                borderColor: "hsl(var(--border))",
                borderRadius: "var(--radius)",
                fontSize: "12px",
              }}
              labelFormatter={(label) => {
                const item = data.find((d) => d.date === label);
                return `${label} - ${item?.regime}`;
              }}
            />
            {data
              .filter((d) => d.regimeChange)
              .map((d, i) => (
                <ReferenceLine
                  key={i}
                  x={d.date}
                  stroke="hsl(var(--primary))"
                  strokeDasharray="3 3"
                  label={{
                    value: "Regime Change",
                    fill: "hsl(var(--primary))",
                    fontSize: 10,
                    position: "insideTopRight",
                  }}
                />
              ))}
            <Area
              type="monotone"
              dataKey="value"
              stroke="hsl(var(--primary))"
              fill="url(#colorValue)"
              strokeWidth={2}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
      <div className="mt-2">
        <h4 className="text-xs font-medium">Market Performance vs Regimes</h4>
        <div className="mt-2 flex space-x-4">
          <RegimeLegend label="Trending & Liquid" color="favorable" />
          <RegimeLegend label="Choppy & Liquid" color="neutral" />
          <RegimeLegend label="Volatile & Illiquid" color="risk" />
        </div>
      </div>
    </div>
  );
}

function RegimeLegend({ label, color }: { label: string; color: "favorable" | "neutral" | "risk" }) {
  return (
    <div className="flex items-center gap-1.5">
      <div
        className={cn(
          "h-3 w-3 rounded-full",
          color === "favorable" && "bg-regime-favorable",
          color === "neutral" && "bg-regime-neutral",
          color === "risk" && "bg-regime-risk"
        )}
      />
      <span className="text-xs text-muted-foreground">{label}</span>
    </div>
  );
}
