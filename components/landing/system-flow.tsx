import { motion } from "framer-motion"
import { FileCode, Settings2, Network, Brain, BarChart3, Shield } from "lucide-react"

const flowSteps = [
  {
    title: "Verilog RTL Input",
    description: "Start with RTL Verilog (.v) design files from the TrustHub benchmark suite.",
    icon: FileCode,
    gradient: "from-amber-500 via-yellow-500 to-amber-400",
    shadowColor: "shadow-amber-500/25",
  },
  {
    title: "Logic Synthesis",
    description: "Convert RTL to gate-level netlists using Yosys synthesis.",
    icon: Settings2,
    gradient: "from-yellow-500 via-amber-500 to-orange-500",
    shadowColor: "shadow-yellow-500/25",
  },
  {
    title: "Graph Construction",
    description: "Build DAGs where nodes are logic gates and edges are signal connections.",
    icon: Network,
    gradient: "from-orange-500 via-amber-500 to-yellow-500",
    shadowColor: "shadow-orange-500/25",
  },
  {
    title: "Feature Extraction",
    description: "Extract gate type, fan-in/out, logic depth, signal probability features.",
    icon: BarChart3,
    gradient: "from-amber-600 via-yellow-500 to-amber-400",
    shadowColor: "shadow-amber-600/25",
  },
  {
    title: "GNN Training",
    description: "Train masked autoencoder GNN (GraphSAGE) on clean circuits only.",
    icon: Brain,
    gradient: "from-yellow-600 via-amber-500 to-orange-400",
    shadowColor: "shadow-yellow-600/25",
  },
  {
    title: "Anomaly Detection",
    description: "Flag high reconstruction error nodes as potential Trojan insertions.",
    icon: Shield,
    gradient: "from-orange-500 via-amber-600 to-yellow-500",
    shadowColor: "shadow-orange-500/25",
  },
]

export function SystemFlow() {
  return (
    <section className="py-20 sm:py-24 lg:py-32">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="text-center"
      >
        <h2 className="text-3xl font-bold tracking-tight text-foreground sm:text-4xl lg:text-5xl">
          Detection{" "}
          <span className="bg-gradient-to-r from-amber-500 via-yellow-500 to-orange-500 bg-clip-text text-transparent">
            Pipeline
          </span>
        </h2>
        <p className="mx-auto mt-6 max-w-2xl text-lg leading-8 text-muted-foreground">
          End-to-end self-supervised pipeline for hardware Trojan detection.
        </p>
      </motion.div>

      <div className="mt-16 grid gap-8 lg:grid-cols-3">
        {flowSteps.map((step, index) => (
          <motion.div
            key={step.title}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: index * 0.1 }}
            className="relative group"
          >
            <div className={`h-full rounded-2xl p-1 transition-all duration-300 bg-gradient-to-br ${step.gradient} opacity-75 hover:opacity-100 hover:scale-[1.02] hover:-translate-y-1`}>
              <div className="h-full rounded-xl bg-background/90 p-6 backdrop-blur-xl">
                <div className="flex items-center gap-3">
                  <div className={`size-14 rounded-lg bg-gradient-to-br ${step.gradient} flex items-center justify-center ${step.shadowColor} shadow-lg transition-shadow duration-300 group-hover:shadow-xl`}>
                    <step.icon className="size-7 text-white" />
                  </div>
                  <span className="text-sm font-mono text-muted-foreground">Step {index + 1}</span>
                </div>
                <h3 className="mt-4 text-xl font-semibold text-foreground">{step.title}</h3>
                <p className="mt-2 text-muted-foreground leading-relaxed">{step.description}</p>
              </div>
            </div>
          </motion.div>
        ))}
      </div>
    </section>
  )
}