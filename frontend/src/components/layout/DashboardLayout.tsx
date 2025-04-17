
import { cn } from "@/lib/utils";
import { ReactNode } from "react";
import { LayoutDashboard, TrendingUp, Bell, History, Settings, User } from "lucide-react";

interface DashboardLayoutProps {
  children: ReactNode;
  className?: string;
}

export function DashboardLayout({ children, className }: DashboardLayoutProps) {
  return (
    <div className={cn("min-h-screen bg-background", className)}>
      <div className="flex min-h-screen">
        <Sidebar />
        <main className="flex-1 overflow-y-auto">
          <div className="container py-6 px-4 md:px-6">
            {children}
          </div>
        </main>
      </div>
    </div>
  );
}

function Sidebar() {
  return (
    <aside className="hidden md:flex md:w-64 flex-col bg-sidebar h-screen sticky top-0 border-r border-border/40">
      <div className="flex h-14 items-center border-b border-border/40 px-4">
        <h2 className="text-lg font-semibold text-primary">MarketPulse</h2>
      </div>
      <nav className="flex-1 p-4">
        <ul className="space-y-2">
          <SidebarItem icon={<LayoutDashboard size={18} />} active>Dashboard</SidebarItem>
          <SidebarItem icon={<TrendingUp size={18} />}>Strategies</SidebarItem>
          <SidebarItem icon={<Bell size={18} />}>Alerts</SidebarItem>
          <SidebarItem icon={<History size={18} />}>History</SidebarItem>
          <SidebarItem icon={<Settings size={18} />}>Settings</SidebarItem>
        </ul>
      </nav>
      <div className="border-t border-border/40 p-4">
        <div className="flex items-center gap-3">
          <div className="flex h-8 w-8 items-center justify-center rounded-full bg-secondary text-primary">
            <User size={16} />
          </div>
          <div className="text-sm">
            <p className="font-medium">User</p>
            <p className="text-muted-foreground">Pro Plan</p>
          </div>
        </div>
      </div>
    </aside>
  );
}

function SidebarItem({ 
  children, 
  icon, 
  active 
}: { 
  children: ReactNode; 
  icon?: ReactNode;
  active?: boolean 
}) {
  return (
    <li>
      <a
        href="#"
        className={cn(
          "flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium",
          active
            ? "bg-accent/50 text-primary"
            : "text-muted-foreground hover:bg-accent/30 hover:text-foreground"
        )}
      >
        {icon && <span className={active ? "text-primary" : "text-muted-foreground"}>{icon}</span>}
        {children}
      </a>
    </li>
  );
}
