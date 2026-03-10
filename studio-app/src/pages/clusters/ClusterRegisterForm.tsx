import { useState } from "react";
import { CommandFormPanel } from "../../components/shared/CommandFormPanel";
import { FormField } from "../../components/shared/FormField";
import { useCrucible } from "../../context/CrucibleContext";
import { startCrucibleCommand, getCrucibleCommandStatus } from "../../api/studioApi";
import type { ClusterConfig } from "../../types/remote";

interface ClusterRegisterFormProps {
  onRegistered: () => void;
  editCluster?: ClusterConfig;
}

export function ClusterRegisterForm({ onRegistered, editCluster }: ClusterRegisterFormProps) {
  const isEdit = !!editCluster;
  const { dataRoot } = useCrucible();
  const [name, setName] = useState(editCluster?.name ?? "");
  const [host, setHost] = useState(editCluster?.host ?? "");
  const [user, setUser] = useState(editCluster?.user ?? "");
  const [sshKey, setSshKey] = useState("");
  const [password, setPassword] = useState("");
  const [partition, setPartition] = useState(editCluster?.defaultPartition ?? "");
  const [moduleLoads, setModuleLoads] = useState(editCluster?.moduleLoads.join(", ") ?? "");
  const [pythonPath, setPythonPath] = useState(editCluster?.pythonPath ?? "python3");
  const [workspace, setWorkspace] = useState(editCluster?.remoteWorkspace ?? "~/crucible-jobs");
  const [validate, setValidate] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [output, setOutput] = useState("");

  const missing: string[] = [];
  if (!name) missing.push("Cluster Name");
  if (!host) missing.push("Host");
  if (!user) missing.push("User");

  async function handleSubmit() {
    setIsRunning(true);
    setError(null);
    setOutput("");
    try {
      const args = [
        "remote", "register-cluster",
        "--name", name,
        "--host", host,
        "--user", user,
      ];
      if (sshKey) args.push("--ssh-key", sshKey);
      if (password) args.push("--password", password);
      if (partition) args.push("--partition", partition);
      if (moduleLoads) args.push("--module-loads", moduleLoads);
      args.push("--python-path", pythonPath);
      args.push("--remote-workspace", workspace);
      if (validate) args.push("--validate");

      const { task_id } = await startCrucibleCommand(dataRoot, args);
      // Poll until the background task completes
      let status = await getCrucibleCommandStatus(task_id);
      while (status.status === "running") {
        await new Promise((r) => setTimeout(r, 500));
        status = await getCrucibleCommandStatus(task_id);
      }
      if (status.status === "failed") {
        const stderr = status.stderr || "Unknown error";
        setError(stderr.split("\n").filter(Boolean).pop() || stderr);
      } else {
        setOutput(status.stdout || `Cluster '${name}' ${isEdit ? "updated" : "registered"} successfully.`);
        onRegistered();
      }
    } catch (err) {
      setError(String(err));
    } finally {
      setIsRunning(false);
    }
  }

  return (
    <CommandFormPanel
      title={isEdit ? "Edit Cluster" : "Register Cluster"}
      missing={missing}
      isRunning={isRunning}
      submitLabel={isEdit ? "Save" : "Register"}
      runningLabel={isEdit ? "Saving..." : "Registering..."}
      onSubmit={handleSubmit}
      error={error}
      output={output}
    >
      <FormField label="Cluster Name" required>
        <input
          className="input"
          value={name}
          onChange={(e) => setName(e.currentTarget.value)}
          placeholder="my-hpc"
          readOnly={isEdit}
        />
      </FormField>

      <FormField label="Host" required hint="SSH hostname or ~/.ssh/config alias">
        <input
          className="input"
          value={host}
          onChange={(e) => setHost(e.currentTarget.value)}
          placeholder="hpc.university.edu"
        />
      </FormField>

      <FormField label="User" required>
        <input
          className="input"
          value={user}
          onChange={(e) => setUser(e.currentTarget.value)}
          placeholder="jdoe"
        />
      </FormField>

      <FormField label="SSH Key" hint={isEdit ? "Leave blank to keep current key" : "Path to private key (optional, uses ssh-agent by default)"}>
        <input
          className="input"
          value={sshKey}
          onChange={(e) => setSshKey(e.currentTarget.value)}
          placeholder="~/.ssh/id_rsa"
        />
      </FormField>

      <FormField label="Password" hint={isEdit ? "Leave blank to keep current password" : "Optional — only needed if key-based auth is not set up"}>
        <input
          className="input"
          type="password"
          value={password}
          onChange={(e) => setPassword(e.currentTarget.value)}
          placeholder=""
        />
      </FormField>

      <FormField label="Default Partition">
        <input
          className="input"
          value={partition}
          onChange={(e) => setPartition(e.currentTarget.value)}
          placeholder="gpu"
        />
      </FormField>

      <FormField label="Module Loads" hint="Comma-separated (e.g. module load cuda/12.1)">
        <input
          className="input"
          value={moduleLoads}
          onChange={(e) => setModuleLoads(e.currentTarget.value)}
          placeholder="module load cuda/12.1, module load python/3.11"
        />
      </FormField>

      <FormField label="Python Path">
        <input
          className="input"
          value={pythonPath}
          onChange={(e) => setPythonPath(e.currentTarget.value)}
        />
      </FormField>

      <FormField label="Remote Workspace">
        <input
          className="input"
          value={workspace}
          onChange={(e) => setWorkspace(e.currentTarget.value)}
        />
      </FormField>

      <FormField label="Validate on Register">
        <label className="flex-row">
          <input
            type="checkbox"
            checked={validate}
            onChange={(e) => setValidate(e.currentTarget.checked)}
          />
          <span>Run validation checks after registration</span>
        </label>
      </FormField>
    </CommandFormPanel>
  );
}
