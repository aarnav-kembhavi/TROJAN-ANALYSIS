export interface TrojanPrediction {
  status: string;
  prediction: string;
  trojan_probability: number;
  max_node_probability: number;
  top_suspicious_nodes: Array<{
    node_id: number;
    probability: number;
  }>;
  node_probabilities?: number[];
  metadata: {
    circuit_name: string;
    node_count: number;
    edge_count: number;
    family: string;
    [key: string]: any;
  };
  artifacts: {
    node_scores_csv: string;
    embedding_shape: number[];
  };
}

export interface HealthCheck {
  status: string;
  app_name: string;
  model_loaded: boolean;
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000';

export const trojanApi = {
  async getHealth(): Promise<HealthCheck> {
    const response = await fetch(`${API_BASE_URL}/api/v1/health`);
    if (!response.ok) {
      throw new Error('Failed to fetch health status');
    }
    return response.json();
  },

  async predict(
    files: File[]
  ): Promise<TrojanPrediction> {
    const formData = new FormData();
    files.forEach(file => {
      formData.append('files', file);
    });

    const response = await fetch(`${API_BASE_URL}/api/v1/predict`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(errorData.detail || 'Failed to get prediction');
    }

    return response.json();
  },

  getArtifactUrl(path: string): string {
    // The backend returns an absolute path or a path relative to the backend root.
    // We need to convert it to a URL served by the backend's static mount.
    // Based on backend/main.py: app.mount("/static", StaticFiles(directory="static"), name="static")
    // and backend/app/core/config.py: RESULTS_DIR = "./static/results"
    // The path in artifacts.node_scores_csv will be something like "/.../backend/static/results/uuid_inference_results.csv"
    
    const fileName = path.split('/').pop();
    return `${API_BASE_URL}/static/results/${fileName}`;
  }
};
