
import { Menu } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import { cn } from "@/lib/utils";
import { ReactNode } from "react";
import { LayoutDashboard, TrendingUp, Bell, History, Settings } from "lucide-react";

export function MobileNav() {
  return (
    <Sheet>
      <SheetTrigger asChild>
        <Button
          variant="ghost"
          size="icon"
          className="md:hidden"
          aria-label="Open menu"
        >
          <Menu className="h-5 w-5" />
        </Button>
      </SheetTrigger>
      <SheetContent side="left" className="w-72">
        <SheetHeader className="border-b pb-4 mb-4">
          <SheetTitle className="text-primary">MarketPulse</SheetTitle>
        </SheetHeader>
        <nav>
          <ul className="space-y-3">
            <NavItem 
              href="/" 
              icon={<LayoutDashboard size={18} />}
              active
            >
              Dashboard
            </NavItem>
            <NavItem 
              href="/strategies" 
              icon={<TrendingUp size={18} />}
            >
              Strategies
            </NavItem>
            <NavItem 
              href="/alerts" 
              icon={<Bell size={18} />}
            >
              Alerts
            </NavItem>
            <NavItem 
              href="/history" 
              icon={<History size={18} />}
            >
              History
            </NavItem>
            <NavItem 
              href="/settings" 
              icon={<Settings size={18} />}
            >
              Settings
            </NavItem>
          </ul>
        </nav>
      </SheetContent>
    </Sheet>
  );
}

interface NavItemProps {
  href: string;
  children: ReactNode;
  icon?: ReactNode;
  active?: boolean;
}

function NavItem({ href, children, icon, active }: NavItemProps) {
  return (
    <li>
      <a
        href={href}
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
