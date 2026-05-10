import { motion } from "framer-motion"
import { ArrowRight } from "lucide-react"

const features = [
  {
    title: "Masked Autoencoder GNN",
    description: "The core model uses a GraphSAGE encoder with an MLP decoder. During training, node features are randomly masked and the model learns to reconstruct them from graph context — capturing normal circuit structure.",
  },
  {
    title: "Anomaly Scoring & Localisation",
    description: "At inference, per-node reconstruction error (MSE) serves as an anomaly score. Nodes with high error indicate structural deviations — potential Trojan insertions are localised at the gate level.",
  },
  {
    title: "One-Class Detection Paradigm",
    description: "Training uses only clean (golden) circuits. The model learns P(normal circuit behavior) and flags deviations. No Trojan data is ever seen during training, making the approach truly unsupervised.",
  },
]

export function FeatureDetails() {
  return (
    <section className="py-20 sm:py-24 lg:py-32">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="text-center"
      >
        <h2 className="text-3xl font-bold tracking-tight text-foreground sm:text-4xl lg:text-5xl">
          Core{" "}
          <span className="bg-gradient-to-r from-amber-500 via-yellow-500 to-orange-500 bg-clip-text text-transparent">
            Methodology
          </span>
        </h2>
        <p className="mx-auto mt-6 max-w-2xl text-lg leading-8 text-muted-foreground">
          Understand the key innovations powering self-supervised anomaly localisation for hardware Trojan detection.
        </p>
      </motion.div>

      <div className="mt-16 grid grid-cols-1 gap-16 sm:gap-24">
        {features.map((feature, index) => (
          <motion.div
            key={feature.title}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: index * 0.1 }}
            className={`flex flex-col gap-8 lg:items-center ${
              index % 2 === 0 ? 'lg:flex-row' : 'lg:flex-row-reverse'
            }`}
          >
            {/* Text Content */}
            <div className="flex-1 space-y-4">
              <motion.div
                initial={{ opacity: 0, x: index % 2 === 0 ? -20 : 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.5, delay: 0.3 }}
              >
                <h3 className="text-2xl font-bold tracking-tight text-foreground sm:text-3xl">
                  {feature.title}
                </h3>
                <p className="mt-4 text-lg leading-8 text-muted-foreground">
                  {feature.description}
                </p>
                <div className="mt-6">
                  <button className="group inline-flex items-center gap-2 text-sm font-semibold text-primary">
                    Learn more
                    <ArrowRight className="size-4 transition-transform group-hover:translate-x-1" />
                  </button>
                </div>
              </motion.div>
            </div>

            {/* Visual */}
            <div className="flex-1">
              <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.5, delay: 0.4 }}
                className="relative rounded-2xl bg-gradient-to-b from-muted/50 to-muted p-2 ring-1 ring-foreground/10 backdrop-blur-3xl dark:from-muted/30 dark:to-background/80"
              >
                <div className="flex items-center justify-center rounded-xl bg-gradient-to-br from-amber-950/10 via-background to-yellow-950/10 min-h-[250px] p-6">
                  {index === 0 && (
                    <div className="text-center space-y-3">
                      <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-amber-500/10 border border-amber-500/20 text-amber-600 dark:text-amber-400 text-sm font-mono">
                        GraphSAGE → MLP Decoder
                      </div>
                      <p className="text-xs text-muted-foreground max-w-xs">Masked features → Neighborhood aggregation → Reconstruction</p>
                    </div>
                  )}
                  {index === 1 && (
                    <div className="text-center space-y-3">
                      <div className="flex items-center gap-4 text-sm font-mono">
                        <span className="px-3 py-1.5 rounded bg-green-500/10 text-green-600 dark:text-green-400 border border-green-500/20">Low Error ✓</span>
                        <span className="px-3 py-1.5 rounded bg-red-500/10 text-red-600 dark:text-red-400 border border-red-500/20">High Error ⚠</span>
                      </div>
                      <p className="text-xs text-muted-foreground max-w-xs">error(node) = ||predicted - actual||²</p>
                    </div>
                  )}
                  {index === 2 && (
                    <div className="text-center space-y-3">
                      <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-amber-500/10 border border-amber-500/20 text-amber-600 dark:text-amber-400 text-sm font-mono">
                        Train: Clean Only → Test: Clean + Trojan
                      </div>
                      <p className="text-xs text-muted-foreground max-w-xs">One-class learning: model never sees Trojans</p>
                    </div>
                  )}
                </div>
                <div className="absolute inset-0 rounded-2xl bg-gradient-to-br from-amber-500/20 via-transparent to-orange-500/20 opacity-0 transition-opacity duration-300 hover:opacity-100" />
              </motion.div>
            </div>
          </motion.div>
        ))}
      </div>
    </section>
  )
} 