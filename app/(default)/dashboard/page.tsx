import { PageHeader } from '@/components/ui/page-header';
import { TrojanAnalysis } from '@/components/dashboard/trojan-analysis';

export default async function DashboardPage() {

  return (
    <>
      <PageHeader
        title="Dashboard"
        description="Overview of hardware Trojan detection analysis and results."
      >
      </PageHeader>
      <div className="mt-8 space-y-12">
        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
          <div className="rounded-2xl border border-border/50 bg-gradient-to-b from-muted/50 to-muted p-6">
            <h3 className="text-lg font-semibold text-foreground">Circuits Analyzed</h3>
            <p className="mt-2 text-3xl font-bold text-amber-500">128</p>
            <p className="mt-1 text-sm text-muted-foreground">TrustHub benchmarks</p>
          </div>
          <div className="rounded-2xl border border-border/50 bg-gradient-to-b from-muted/50 to-muted p-6">
            <h3 className="text-lg font-semibold text-foreground">Anomalies Detected</h3>
            <p className="mt-2 text-3xl font-bold text-amber-500">12</p>
            <p className="mt-1 text-sm text-muted-foreground">Flagged nodes</p>
          </div>
          <div className="rounded-2xl border border-border/50 bg-gradient-to-b from-muted/50 to-muted p-6">
            <h3 className="text-lg font-semibold text-foreground">Model Status</h3>
            <p className="mt-2 text-3xl font-bold text-green-500">Active</p>
            <p className="mt-1 text-sm text-muted-foreground">GNN masked autoencoder</p>
          </div>
        </div>

        <TrojanAnalysis />
      </div>
    </>
  );
}
