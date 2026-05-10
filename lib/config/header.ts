import { LucideIcon } from "lucide-react"

export interface HeaderLink {
  href: string
  label: string
  icon?: LucideIcon
  description?: string
}

export interface HeaderConfig {
  brand: {
    title: string
    icon: string
  }
  navigationLinks: HeaderLink[]
}

export const headerConfig: HeaderConfig = {
  brand: {
    title: "HT Detector",
    icon: "/next.svg"
  },
  navigationLinks: [
    {
      href: "/",
      label: "Home"
    },
    {
      href: "/dashboard",
      label: "Dashboard"
    },
    {
      href: "/chat",
      label: "Chat"
    }
  ]
}