
import { DashboardLayout } from "@/components/layout/DashboardLayout";
import { MarketRegimeIndicator } from "@/components/dashboard/MarketRegimeIndicator";
import { TimelineChart } from "@/components/dashboard/TimelineChart";
import { KeyMetricsPanel } from "@/components/dashboard/KeyMetricsPanel";
import { StrategyRecommendations } from "@/components/dashboard/StrategyRecommendations";
import { AlertsSystem } from "@/components/dashboard/AlertsSystem";
import { HistoricalAnalysis } from "@/components/dashboard/HistoricalAnalysis";
import { MobileNav } from "@/components/dashboard/MobileNav";

const Index = () => {
  return (
    <DashboardLayout>
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-semibold">Market Analytics Dashboard</h1>
        <MobileNav />
      </div>
      
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        {/* Current Market Regime */}
        <MarketRegimeIndicator 
          regime="Volatile & Illiquid"
          status="risk"
          className="col-span-full md:col-span-1"
          style={{ animationDelay: '0s' }}
        />
        
        {/* Timeline Chart */}
        <TimelineChart 
          className="col-span-full md:col-span-2"
        />
        
        {/* Key Metrics */}
        <KeyMetricsPanel 
          className="col-span-full animate-fade-in"
          style={{ animationDelay: '0.1s' }}
        />
        
        {/* Strategy Recommendations */}
        <StrategyRecommendations
          className="col-span-full md:col-span-1 lg:row-span-2 animate-fade-in"
          style={{ animationDelay: '0.2s' }}
        />
        
        {/* Alerts System */}
        <AlertsSystem
          className="col-span-full md:col-span-1 animate-fade-in"
          style={{ animationDelay: '0.3s' }}
        />
        
        {/* Historical Analysis */}
        <HistoricalAnalysis
          className="col-span-full lg:col-span-2 animate-fade-in"
          style={{ animationDelay: '0.4s' }}
        />
      </div>
    </DashboardLayout>
  );
};

export default Index;
