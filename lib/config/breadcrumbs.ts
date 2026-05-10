

export const routes = {
    "/": {
      name: "Home",
      path: "/",
    },
    "/dashboard": {
      name: "Dashboard",
      path: "/dashboard",
      description: "Detection overview and results",
    },
    "/chat": {
      name: "Chat",
      path: "/chat",
      description: "AI-assisted circuit analysis",
    },
    "/settings": {
      name: "Settings",
      path: "/settings",
      description: "System configuration",
      subRoutes: {
        "/settings/analysis": {
          name: "Analysis Configuration",
          path: "/settings/analysis",
        },
        "/settings/thresholds": {
          name: "Detection Thresholds",
          path: "/settings/thresholds",
        }
      }
    }
  }