export interface FooterLink {
  href: string
  label: string
}

export interface FooterSection {
  title: string
  links: FooterLink[]
}

export interface FooterConfig {
  brand: {
    title: string
    description: string
  }
  sections: FooterSection[]
  copyright: string
}

export const footerConfig: FooterConfig = {
  brand: {
    title: "HT Detector",
    description: "Self-supervised anomaly localisation for hardware Trojans using Graph Neural Networks."
  },
  sections: [
    {
      title: "Platform",
      links: [
        { href: "/chat", label: "Chat" },
      ]
    },
    {
      title: "Research",
      links: [
        { href: "#", label: "TrustHub Benchmarks" },
        { href: "#", label: "Documentation" },
      ]
    }
  ],
  copyright: `© ${new Date().getFullYear()} HT Detector. All rights reserved.`
}
