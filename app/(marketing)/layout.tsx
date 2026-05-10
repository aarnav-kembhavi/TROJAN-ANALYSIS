import { Metadata } from "next"
import { Header } from "@/components/global/header"
import { headerConfig } from "@/lib/config/header"
import Footer from "@/components/global/footer"
import { footerConfig } from "@/lib/config/footer"

export const metadata: Metadata = {
  title: "HT Detector — Self-Supervised Hardware Trojan Detection",
  description: "Self-supervised anomaly localisation for hardware Trojans using Graph Neural Networks",
}

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
        <main>
            <Header config={headerConfig} />
            {children}
            <Footer config={footerConfig} />
        </main>
  )
} 