"use client"

import { ArrowRight, ArrowRightIcon } from "lucide-react"
import Link from "next/link"
import { BlurFade } from "@/components/magicui/blur-fade"
import { BorderBeam } from "@/components/magicui/border-beam"
import { ShimmerButton } from "@/components/magicui/shimmer-button"
import { AnimatedShinyText } from "@/components/magicui/animated-shiny-text"
import { cn } from "@/lib/utils"

export function Hero() {
  return (
    <section className="py-20 sm:py-24 lg:py-32">
      <div className="group relative mx-auto flex justify-center">
        <BlurFade delay={0.25} inView>
          <Link
            href="https://github.com"
            target="_blank"
            className="group rounded-full border border-black/5 bg-neutral-100 text-base text-white transition-all ease-in hover:cursor-pointer hover:bg-neutral-200 dark:border-white/5 dark:bg-neutral-900 dark:hover:bg-neutral-800"
          >
            <AnimatedShinyText className="inline-flex items-center justify-center px-4 py-1 transition ease-out hover:text-neutral-600 hover:duration-300 hover:dark:text-neutral-400">
              <span>🔒 Self-Supervised Anomaly Detection</span>
              <ArrowRightIcon className="ml-1 size-3 transition-transform duration-300 ease-in-out group-hover:translate-x-0.5" />
            </AnimatedShinyText>
          </Link>
        </BlurFade>
      </div>

      <div className="mt-10 text-center">
        <BlurFade delay={0.5} inView>
          <h1 className="mx-auto max-w-4xl text-4xl font-bold tracking-tight text-foreground sm:text-6xl lg:text-7xl">
            Hardware Trojan{" "}
            <span className="bg-gradient-to-r from-amber-500 via-yellow-500 to-orange-500 bg-clip-text text-transparent">
              Detection
            </span>{" "}
            Framework
          </h1>
        </BlurFade>
        <BlurFade delay={0.75} inView>
          <p className="mx-auto mt-6 max-w-2xl text-lg leading-8 text-muted-foreground sm:mt-8">
            Self-supervised anomaly localisation for hardware trojans using Graph Neural Networks.
            Learn normal circuit behavior and detect malicious modifications through reconstruction-based anomaly scoring.
          </p>
        </BlurFade>
      </div>

      <div className="mt-8 flex items-center justify-center gap-4 sm:mt-10">
        <BlurFade delay={1.0} inView>
          <Link href="/dashboard">
            <ShimmerButton
              className="flex items-center gap-2 px-6 py-3 text-base sm:text-lg"
              background="linear-gradient(to right, #D97706, #F59E0B)"
            >
              <span className="whitespace-pre-wrap text-center font-medium leading-none tracking-tight text-white">
                Try Analysis
              </span>
              <ArrowRight className="w-4 h-4 transition-transform group-hover:translate-x-1 sm:w-5 sm:h-5" />
            </ShimmerButton>
          </Link>
        </BlurFade>
        <BlurFade delay={1.25} inView>
          <Link href="https://github.com" target="_blank">
            <div>
              <ShimmerButton
                className="flex items-center gap-2 px-6 py-3 text-base sm:text-lg"
                background="linear-gradient(to right, #334155, #0f172a)"
              >
                <span className="whitespace-pre-wrap text-center font-medium leading-none tracking-tight text-white">
                  View Research
                </span>
                <ArrowRight className="w-4 h-4 transition-transform group-hover:translate-x-1 sm:w-5 sm:h-5" />
              </ShimmerButton>
            </div>
          </Link>
        </BlurFade>
      </div>

      <div className="relative mx-auto mt-16 sm:mt-20 lg:mt-24">
        <BlurFade delay={1.5} inView>
          <div className="relative rounded-2xl bg-gradient-to-b from-muted/50 to-muted p-2 ring-1 ring-foreground/10 backdrop-blur-3xl dark:from-muted/30 dark:to-background/80">
            {/* Pipeline visualization placeholder */}
            <div className="flex items-center justify-center rounded-xl bg-gradient-to-br from-amber-950/20 via-background to-yellow-950/20 shadow-2xl ring-1 ring-foreground/10 transition-all duration-300 min-h-[320px] sm:min-h-[400px]">
              <div className="text-center p-8 max-w-3xl">
                <div className="flex items-center justify-center gap-3 sm:gap-6 text-sm sm:text-base font-mono text-muted-foreground flex-wrap">
                  <span className="px-3 py-1.5 rounded-lg bg-amber-500/10 text-amber-600 dark:text-amber-400 border border-amber-500/20">Verilog</span>
                  <span className="text-amber-500">→</span>
                  <span className="px-3 py-1.5 rounded-lg bg-amber-500/10 text-amber-600 dark:text-amber-400 border border-amber-500/20">Netlist</span>
                  <span className="text-amber-500">→</span>
                  <span className="px-3 py-1.5 rounded-lg bg-amber-500/10 text-amber-600 dark:text-amber-400 border border-amber-500/20">Graph</span>
                  <span className="text-amber-500">→</span>
                  <span className="px-3 py-1.5 rounded-lg bg-amber-500/10 text-amber-600 dark:text-amber-400 border border-amber-500/20">GNN</span>
                  <span className="text-amber-500">→</span>
                  <span className="px-3 py-1.5 rounded-lg bg-amber-500/10 text-amber-600 dark:text-amber-400 border border-amber-500/20">Anomaly Score</span>
                </div>
                <p className="mt-6 text-muted-foreground text-sm">End-to-End Hardware Trojan Detection Pipeline</p>
              </div>
            </div>
            <BorderBeam size={250} duration={12} delay={9} />
          </div>
        </BlurFade>
      </div>
    </section>
  )
}