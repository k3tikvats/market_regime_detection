
import { 
  AreaChart, 
  Area, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  ReferenceLine 
} from 'recharts';
import { cn } from '@/lib/utils';

// Sample timeline data
const data = [
  { date: 'Apr 12', value: 0.8, regime: 'Trending & Liquid' },
  { date: 'Apr 13', value: 0.7, regime: 'Trending & Liquid' },
  { date: 'Apr 14', value: 0.9, regime: 'Trending & Liquid' },
  { date: 'Apr 15', value: 0.5, regime: 'Choppy & Liquid' },
  { date: 'Apr 16', value: 0.3, regime: 'Choppy & Liquid' },
  { date: 'Apr 17', value: -0.2, regime: 'Volatile & Illiquid' },
  { date: 'Apr 18', value: -0.5, regime: 'Volatile & Illiquid' },
  { date: 'Apr 19', value: -0.3, regime: 'Volatile & Illiquid' },
  { date: 'Apr 20', value: 0.1, regime: 'Trending & Illiquid' },
  { date: 'Apr 21', value: 0.4, regime: 'Trending & Illiquid' },
];

// Regime transitions
const transitions = [
  { date: 'Apr 15', label: 'Regime Change' },
  { date: 'Apr 17', label: 'Regime Change' },
  { date: 'Apr 20', label: 'Regime Change' },
];

interface TimelineChartProps {
  className?: string;
  style?: React.CSSProperties;
}

export function TimelineChart({ className, style }: TimelineChartProps) {
  return (
    <div className={cn("rounded-lg border p-6 animate-fade-in", className)} style={style}>
      <div className="mb-4 flex flex-col space-y-1.5">
        <h3 className="text-sm font-medium text-muted-foreground">Regime Timeline</h3>
        <p className="text-xs text-muted-foreground">Last 10 days</p>
      </div>
      <div className="h-[200px]">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart
            data={data}
            margin={{ top: 20, right: 30, left: 0, bottom: 0 }}
          >
            <defs>
              <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="hsl(var(--primary))" stopOpacity={0.3} />
                <stop offset="95%" stopColor="hsl(var(--primary))" stopOpacity={0} />
              </linearGradient>
            </defs>
            <XAxis 
              dataKey="date" 
              tick={{ fontSize: 10, fill: 'hsl(var(--muted-foreground))' }}
              axisLine={{ stroke: 'hsl(var(--border))' }}
              tickLine={false}
            />
            <YAxis 
              hide 
              domain={[-1, 1]} 
            />
            <CartesianGrid 
              strokeDasharray="3 3"
              stroke="hsl(var(--border))" 
              opacity={0.3}
            />
            <Tooltip
              contentStyle={{ 
                backgroundColor: 'hsl(var(--card))', 
                borderColor: 'hsl(var(--border))',
                borderRadius: 'var(--radius)',
                fontSize: '12px'
              }}
              itemStyle={{ color: 'hsl(var(--foreground))' }}
              labelStyle={{ color: 'hsl(var(--foreground))' }}
            />
            {transitions.map((t, i) => (
              <ReferenceLine
                key={i}
                x={t.date}
                stroke="hsl(var(--primary))"
                strokeDasharray="3 3"
                label={{
                  value: t.label,
                  fill: 'hsl(var(--primary))',
                  fontSize: 10,
                  position: 'insideTopRight',
                }}
              />
            ))}
            <Area 
              type="monotone" 
              dataKey="value" 
              stroke="hsl(var(--primary))" 
              strokeWidth={2}
              fill="url(#colorValue)" 
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
