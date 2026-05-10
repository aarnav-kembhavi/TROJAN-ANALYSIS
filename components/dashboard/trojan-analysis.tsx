'use client';

import React, { useState, useEffect } from 'react';
import { 
  Upload, 
  Search, 
  FileText, 
  AlertTriangle, 
  CheckCircle2, 
  Download, 
  Loader2, 
  Activity,
  Info
} from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent, CardDescription, CardFooter } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { 
  Table, 
  TableBody, 
  TableCell, 
  TableHead, 
  TableHeader, 
  TableRow 
} from '@/components/ui/table';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { trojanApi, TrojanPrediction, HealthCheck } from '@/lib/api/trojan-api';
import { cn } from '@/lib/utils';

export function TrojanAnalysis() {
  const [featuresFile, setFeaturesFile] = useState<File | null>(null);
  const [edgesFile, setEdgesFile] = useState<File | null>(null);
  const [metaFile, setMetaFile] = useState<File | null>(null);
  
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<TrojanPrediction | null>(null);
  const [health, setHealth] = useState<HealthCheck | null>(null);

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const healthStatus = await trojanApi.getHealth();
        setHealth(healthStatus);
      } catch (err) {
        console.error('Backend health check failed:', err);
      }
    };
    checkHealth();
  }, []);

  const handlePredict = async () => {
    if (!featuresFile || !edgesFile) {
      setError('Please upload both features and edges files.');
      return;
    }

    setIsLoading(true);
    setError(null);
    
    try {
      const prediction = await trojanApi.predict(featuresFile, edgesFile, metaFile || undefined);
      setResult(prediction);
    } catch (err: any) {
      setError(err.message || 'An error occurred during prediction.');
    } finally {
      setIsLoading(false);
    }
  };

  const isHealthy = health?.status === 'healthy' && health?.model_loaded;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold tracking-tight">Trojan Analysis Tool</h2>
        <div className="flex items-center gap-2">
          <Badge variant={isHealthy ? "outline" : "destructive"} className={cn(
            isHealthy ? "bg-green-500/10 text-green-500 border-green-500/20" : ""
          )}>
            <Activity className="mr-1 h-3 w-3" />
            Backend: {health ? (isHealthy ? "Healthy" : "Model Not Loaded") : "Offline"}
          </Badge>
        </div>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        {/* Upload Card */}
        <Card className="border-border/50 bg-gradient-to-b from-muted/50 to-muted">
          <CardHeader>
            <CardTitle className="text-lg">Upload Circuit Data</CardTitle>
            <CardDescription>Upload .npy files for circuit features and edges.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid w-full items-center gap-1.5">
              <Label htmlFor="features">Features File (.npy)</Label>
              <Input 
                id="features" 
                type="file" 
                accept=".npy" 
                onChange={(e) => setFeaturesFile(e.target.files?.[0] || null)}
              />
            </div>
            <div className="grid w-full items-center gap-1.5">
              <Label htmlFor="edges">Edges File (.npy)</Label>
              <Input 
                id="edges" 
                type="file" 
                accept=".npy" 
                onChange={(e) => setEdgesFile(e.target.files?.[0] || null)}
              />
            </div>
            <div className="grid w-full items-center gap-1.5">
              <Label htmlFor="meta">Metadata File (.json, optional)</Label>
              <Input 
                id="meta" 
                type="file" 
                accept=".json" 
                onChange={(e) => setMetaFile(e.target.files?.[0] || null)}
              />
            </div>
          </CardContent>
          <CardFooter>
            <Button 
              onClick={handlePredict} 
              disabled={isLoading || !featuresFile || !edgesFile}
              className="w-full"
            >
              {isLoading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Analyzing...
                </>
              ) : (
                <>
                  <Search className="mr-2 h-4 w-4" />
                  Run Analysis
                </>
              )}
            </Button>
          </CardFooter>
        </Card>

        {/* Prediction Summary Card */}
        {result && (
          <Card className={cn(
            "border-border/50 bg-gradient-to-b",
            result.prediction === "TROJAN DETECTED" 
              ? "from-red-500/10 to-red-500/5 border-red-500/20" 
              : "from-green-500/10 to-green-500/5 border-green-500/20"
          )}>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="text-lg">Analysis Result</CardTitle>
                {result.prediction === "TROJAN DETECTED" ? (
                  <AlertTriangle className="h-5 w-5 text-red-500" />
                ) : (
                  <CheckCircle2 className="h-5 w-5 text-green-500" />
                )}
              </div>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="text-center py-2">
                <div className={cn(
                  "text-3xl font-bold mb-1",
                  result.prediction === "TROJAN DETECTED" ? "text-red-500" : "text-green-500"
                )}>
                  {result.prediction}
                </div>
                <p className="text-sm text-muted-foreground">
                  Circuit: {result.metadata.circuit_name}
                </p>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="rounded-xl border border-border/50 bg-background/50 p-4 text-center">
                  <p className="text-xs text-muted-foreground uppercase tracking-wider mb-1">Trojan Prob.</p>
                  <p className="text-2xl font-bold">{(result.trojan_probability * 100).toFixed(2)}%</p>
                </div>
                <div className="rounded-xl border border-border/50 bg-background/50 p-4 text-center">
                  <p className="text-xs text-muted-foreground uppercase tracking-wider mb-1">Max Node Prob.</p>
                  <p className="text-2xl font-bold">{(result.max_node_probability * 100).toFixed(2)}%</p>
                </div>
              </div>

              <div className="space-y-2">
                <p className="text-sm font-medium">Artifacts</p>
                <Button variant="outline" size="sm" className="w-full" asChild>
                  <a href={trojanApi.getArtifactUrl(result.artifacts.node_scores_csv)} download>
                    <Download className="mr-2 h-4 w-4" />
                    Download Node Scores (CSV)
                  </a>
                </Button>
              </div>
            </CardContent>
          </Card>
        )}
      </div>

      {error && (
        <Alert variant="destructive">
          <AlertTriangle className="h-4 w-4" />
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Suspicious Nodes Table */}
      {result && (
        <Card className="border-border/50 bg-gradient-to-b from-muted/50 to-muted">
          <CardHeader>
            <CardTitle className="text-lg flex items-center">
              <Activity className="mr-2 h-5 w-5 text-amber-500" />
              Top Suspicious Nodes
            </CardTitle>
            <CardDescription>
              Nodes with the highest probability of being part of a Trojan.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Node ID</TableHead>
                  <TableHead>Probability</TableHead>
                  <TableHead className="text-right">Risk Level</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {result.top_suspicious_nodes.map((node) => (
                  <TableRow key={node.node_id}>
                    <TableCell className="font-medium">#{node.node_id}</TableCell>
                    <TableCell>{(node.probability * 100).toFixed(2)}%</TableCell>
                    <TableCell className="text-right">
                      <Badge className={cn(
                        node.probability > 0.8 ? "bg-red-500" : 
                        node.probability > 0.5 ? "bg-amber-500" : "bg-blue-500"
                      )}>
                        {node.probability > 0.8 ? "High" : 
                         node.probability > 0.5 ? "Medium" : "Low"}
                      </Badge>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      )}

      {/* Circuit Metadata */}
      {result && (
        <Card className="border-border/50 bg-gradient-to-b from-muted/50 to-muted">
          <CardHeader>
            <CardTitle className="text-lg flex items-center">
              <Info className="mr-2 h-5 w-5 text-blue-500" />
              Circuit Metadata
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div>
                <p className="text-xs text-muted-foreground uppercase tracking-wider">Family</p>
                <p className="text-sm font-medium">{result.metadata.family}</p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground uppercase tracking-wider">Nodes</p>
                <p className="text-sm font-medium">{result.metadata.node_count}</p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground uppercase tracking-wider">Edges</p>
                <p className="text-sm font-medium">{result.metadata.edge_count}</p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground uppercase tracking-wider">Embeddings</p>
                <p className="text-sm font-medium">{result.artifacts.embedding_shape.join('x')}</p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
