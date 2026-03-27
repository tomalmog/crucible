import { createHashRouter } from "react-router";
import App from "./App";
import { TrainingPage } from "./pages/training/TrainingPage";
import { DatasetsPage } from "./pages/datasets/DatasetsPage";
import { ModelsPage } from "./pages/models/ModelsPage";
import { ChatPage } from "./pages/chat/ChatPage";
import { SettingsPage } from "./pages/settings/SettingsPage";
import { BenchmarksPage } from "./pages/experiments/ExperimentsPage";
import { HubPage } from "./pages/hub/HubPage";
import { CompareChatPage } from "./pages/chat/CompareChatPage";
import { JobsPage } from "./pages/jobs/JobsPage";
import { DocsPage } from "./pages/docs/DocsPage";
import { ClustersPage } from "./pages/clusters/ClustersPage";
import { ResourcesPage } from "./pages/resources/ResourcesPage";
import { InterpPage } from "./pages/interp/InterpPage";
import { UITestPage } from "./pages/ui-test/UITestPage";
import { ExportPage } from "./pages/export/ExportPage";
import { DashboardPage } from "./pages/dashboard/DashboardPage";

export const router = createHashRouter([
  {
    path: "/",
    Component: App,
    children: [
      { index: true, Component: DashboardPage },
      { path: "dashboard", Component: DashboardPage },
      { path: "training", Component: TrainingPage },
      { path: "datasets", Component: DatasetsPage },
      { path: "models", Component: ModelsPage },
      { path: "chat", Component: ChatPage },
      { path: "settings", Component: SettingsPage },
      { path: "benchmarks", Component: BenchmarksPage },
      { path: "hub", Component: HubPage },
      { path: "compare-chat", Component: CompareChatPage },
      { path: "jobs", Component: JobsPage },
      { path: "clusters", Component: ClustersPage },
      { path: "resources", Component: ResourcesPage },
      { path: "interpretability", Component: InterpPage },
      { path: "export", Component: ExportPage },
      { path: "docs", Component: DocsPage },
      { path: "ui-test", Component: UITestPage },
    ],
  },
]);
