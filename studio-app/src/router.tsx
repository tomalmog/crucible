import { createHashRouter } from "react-router";
import App from "./App";
import { TrainingPage } from "./pages/training/TrainingPage";
import { DatasetsPage } from "./pages/datasets/DatasetsPage";
import { ModelsPage } from "./pages/models/ModelsPage";
import { ChatPage } from "./pages/chat/ChatPage";
import { SafetyPage } from "./pages/safety/SafetyPage";
import { DeployPage } from "./pages/deploy/DeployPage";
import { SettingsPage } from "./pages/settings/SettingsPage";
import { ExperimentsPage } from "./pages/experiments/ExperimentsPage";
import { HubPage } from "./pages/hub/HubPage";
import { CompareChatPage } from "./pages/chat/CompareChatPage";
import { JobsPage } from "./pages/jobs/JobsPage";
import { DocsPage } from "./pages/docs/DocsPage";

export const router = createHashRouter([
  {
    path: "/",
    Component: App,
    children: [
      { index: true, Component: TrainingPage },
      { path: "training", Component: TrainingPage },
      { path: "datasets", Component: DatasetsPage },
      { path: "models", Component: ModelsPage },
      { path: "chat", Component: ChatPage },
      { path: "safety", Component: SafetyPage },
      { path: "deploy", Component: DeployPage },
      { path: "settings", Component: SettingsPage },
      { path: "experiments", Component: ExperimentsPage },
      { path: "hub", Component: HubPage },
      { path: "compare-chat", Component: CompareChatPage },
      { path: "jobs", Component: JobsPage },
      { path: "docs", Component: DocsPage },
    ],
  },
]);
