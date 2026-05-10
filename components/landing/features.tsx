"use client"

import { motion } from "framer-motion"
import { Brain, GitBranch, Network, Search, Shield, Cpu } from "lucide-react"
import SectionBadge from "@/components/ui/section-badge"
import { cn } from "@/lib/utils"

const features = [
  {
    title: "Graph Neural Networks",
    info: "Leverage GraphSAGE-based encoders to learn structural representations of gate-level netlists for anomaly detection.",
    icon: Network,
    gradient: "from-amber-500 to-yellow-500",
  },
  {
    title: "Self-Supervised Learning",
    info: "Train exclusively on clean circuits using masked feature reconstruction — no labelled Trojan data required.",
    icon: Brain,
    gradient: "from-orange-500 to-amber-500",
  },
  {
    title: "Node-Level Localisation",
    info: "Identify and flag individual gates exhibiting anomalous behavior through per-node reconstruction error analysis.",
    icon: Search,
    gradient: "from-yellow-500 to-amber-500",
  },
  {
    title: "TrustHub Benchmarks",
    info: "Evaluated on industry-standard TrustHub benchmark circuits including AES and UART families.",
    icon: Shield,
    gradient: "from-amber-600 to-orange-500",
  },
  {
    title: "Automated Pipeline",
    info: "End-to-end flow from Verilog RTL through logic synthesis, graph construction, feature extraction, to anomaly scoring.",
    icon: GitBranch,
    gradient: "from-yellow-600 to-amber-500",
  },
  {
    title: "Hardware-Aware Features",
    info: "Rich node feature vectors including gate type, fan-in/out, logic depth, signal probability, and structural metrics.",
    icon: Cpu,
    gradient: "from-orange-500 to-yellow-600",
  },
]

const container = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1
    }
  }
}

const item = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0 }
}

export function Features() {
  return (
    <section className="py-20 sm:py-24 lg:py-32">
      <div className="text-center">
        <SectionBadge title="Features" />
        <motion.h2 
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
          className="mt-8 text-3xl font-bold tracking-tight text-foreground sm:text-4xl lg:text-5xl"
        >
          Why This Approach Works
        </motion.h2>
        <motion.p 
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="mx-auto mt-6 max-w-2xl text-lg text-muted-foreground"
        >
          Discover the key techniques and innovations behind self-supervised hardware Trojan detection using graph-based anomaly localisation.
        </motion.p>
      </div>

      <motion.div 
        variants={container}
        initial="hidden"
        whileInView="show"
        viewport={{ once: true }}
        className="mt-16 grid grid-cols-1 gap-6 sm:mt-20 sm:grid-cols-2 lg:mt-24 lg:grid-cols-3"
      >
        {features.map((feature) => {
          const Icon = feature.icon;
          return (
            <motion.div
              key={feature.title}
              variants={item}
              className={cn(
                "group relative overflow-hidden rounded-2xl bg-gradient-to-b from-muted/50 to-muted p-8",
                "ring-1 ring-foreground/10 backdrop-blur-xl transition-all duration-300 hover:ring-foreground/20",
                "dark:from-muted/30 dark:to-background/80"
              )}
            >
              <div className="flex items-center gap-4">
                <div className={cn(
                  "flex h-12 w-12 items-center justify-center rounded-xl bg-gradient-to-br",
                  feature.gradient,
                  "ring-1 ring-foreground/10"
                )}>
                  <Icon className="h-6 w-6 text-white" />
                </div>
                <h3 className="text-xl font-semibold text-foreground">
                  {feature.title}
                </h3>
              </div>
              <p className="mt-4 text-muted-foreground">
                {feature.info}
              </p>
              <div className={cn(
                "absolute bottom-0 left-0 right-0 h-1 bg-gradient-to-r",
                feature.gradient,
                "opacity-0 transition-opacity duration-300 group-hover:opacity-100"
              )} />
            </motion.div>
          );
        })}
      </motion.div>
    </section>
  )
} 