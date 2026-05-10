import Icons from "@/components/global/icons";
import { SidebarConfig } from "@/components/global/app-sidebar";

const sidebarConfig: SidebarConfig = {
  brand: {
    title: "HT Detector",
    icon: Icons.shield,
    href: "/"
  },
  sections: [
    {
      label: "Features",
      items: [
        {
          title: "Dashboard",
          href: "/dashboard",
          icon: Icons.shield
        },
        {
          title: "Chat",
          href: "/chat",
          icon: Icons.brain
        },
      ]
    },
  ]
}

export default sidebarConfig