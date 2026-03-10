import { useState } from "react";
import { useCrucibleCommand } from "../../hooks/useCrucibleCommand";
import { useCrucible } from "../../context/CrucibleContext";
import { FormField } from "../../components/shared/FormField";

export function CloudBurstForm() {
  const { dataRoot } = useCrucible();
  const command = useCrucibleCommand();
  const [provider, setProvider] = useState("modal");
  const [gpuType, setGpuType] = useState("a100");
  const [hours, setHours] = useState("2.0");
  const [apiKey, setApiKey] = useState("");
  const [estimate, setEstimate] = useState("");
  const [jobId, setJobId] = useState("");
  const [loading, setLoading] = useState(false);

  async function getEstimate() {
    if (!dataRoot) return;
    setLoading(true);
    const status = await command.run(dataRoot, [
      "cloud", "estimate", "--hours", hours, "--provider", provider, "--gpu-type", gpuType,
    ]);
    if (status.status === "completed" && command.output) {
      setEstimate(command.output);
    }
    setLoading(false);
  }

  async function submitJob() {
    if (!dataRoot || !apiKey.trim()) return;
    setLoading(true);
    const status = await command.run(dataRoot, [
      "cloud", "submit", "--provider", provider, "--api-key", apiKey,
    ]);
    if (status.status === "completed" && command.output) {
      const match = command.output.match(/job_id=(\S+)/);
      if (match) setJobId(match[1]);
    }
    setLoading(false);
  }

  return (
    <div className="panel stack-md">
      <h3>Cloud Training</h3>
      <div className="grid-2">
        <FormField label="Provider">
          <select value={provider} onChange={(e) => setProvider(e.currentTarget.value)}>
            <option value="modal">Modal</option>
            <option value="runpod">RunPod</option>
            <option value="lambda">Lambda</option>
          </select>
        </FormField>
        <FormField label="GPU Type">
          <select value={gpuType} onChange={(e) => setGpuType(e.currentTarget.value)}>
            <option value="a100">A100</option>
            <option value="h100">H100</option>
            <option value="t4">T4</option>
          </select>
        </FormField>
        <FormField label="Estimated Hours">
          <input value={hours} onChange={(e) => setHours(e.currentTarget.value)} />
        </FormField>
        <FormField label="API Key">
          <input type="password" value={apiKey} onChange={(e) => setApiKey(e.currentTarget.value)} placeholder="Provider API key" />
        </FormField>
      </div>
      <div className="row">
        <button className="btn" onClick={() => getEstimate().catch(console.error)} disabled={loading}>
          Get Estimate
        </button>
        <button className="btn btn-primary" onClick={() => submitJob().catch(console.error)} disabled={loading || !apiKey.trim()}>
          Submit Job
        </button>
      </div>
      {estimate && <pre className="console">{estimate}</pre>}
      {jobId && <p className="text-success">Job submitted: {jobId}</p>}
    </div>
  );
}
