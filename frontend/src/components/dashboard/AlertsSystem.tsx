
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Bell, Settings } from "lucide-react";

interface Alert {
  id: string;
  date: string;
  type: "regime-change" | "volatility" | "liquidity";
  message: string;
  status: "favorable" | "neutral" | "risk";
  read: boolean;
}

const recentAlerts: Alert[] = [
  {
    id: "1",
    date: "Today, 09:30 AM",
    type: "regime-change",
    message: "Market regime shifted to Volatile & Illiquid",
    status: "risk",
    read: false,
  },
  {
    id: "2",
    date: "Yesterday, 03:15 PM",
    type: "volatility",
    message: "Volatility spike detected above 30%",
    status: "neutral",
    read: false,
  },
  {
    id: "3",
    date: "Apr 15, 10:45 AM",
    type: "regime-change",
    message: "Market regime shifted to Choppy & Liquid",
    status: "neutral",
    read: true,
  },
  {
    id: "4",
    date: "Apr 12, 02:30 PM",
    type: "liquidity",
    message: "Liquidity conditions improving",
    status: "favorable",
    read: true,
  },
];

interface AlertsSystemProps {
  className?: string;
  style?: React.CSSProperties;
}

export function AlertsSystem({ className, style }: AlertsSystemProps) {
  const unreadCount = recentAlerts.filter((alert) => !alert.read).length;
  
  return (
    <div className={cn("rounded-lg border p-6", className)} style={style}>
      <div className="mb-4 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <h3 className="text-sm font-medium text-muted-foreground">Recent Alerts</h3>
          {unreadCount > 0 && (
            <Badge variant="outline" className="bg-primary/20 text-xs">
              {unreadCount} new
            </Badge>
          )}
        </div>
        <Button variant="outline" size="sm" className="text-xs flex items-center gap-1">
          <Settings size={14} />
          Configure Alerts
        </Button>
      </div>
      <div className="space-y-2">
        {recentAlerts.map((alert) => (
          <AlertItem key={alert.id} alert={alert} />
        ))}
      </div>
    </div>
  );
}

function AlertItem({ alert }: { alert: Alert }) {
  return (
    <div 
      className={cn(
        "flex items-start gap-3 rounded-md p-3",
        !alert.read && "bg-secondary/50",
      )}
    >
      <div 
        className={cn(
          "mt-1 h-2 w-2 rounded-full",
          alert.status === "favorable" && "bg-regime-favorable",
          alert.status === "neutral" && "bg-regime-neutral",
          alert.status === "risk" && "bg-regime-risk",
        )}
      />
      <div className="flex-1">
        <p className={cn("text-sm", !alert.read && "font-medium")}>
          {alert.message}
        </p>
        <div className="mt-1 flex items-center justify-between">
          <span className="text-xs text-muted-foreground">{alert.date}</span>
          <Badge 
            variant="outline" 
            className="text-xs"
          >
            {alert.type === "regime-change" && "Regime Change"}
            {alert.type === "volatility" && "Volatility Alert"}
            {alert.type === "liquidity" && "Liquidity Alert"}
          </Badge>
        </div>
      </div>
    </div>
  );
}
