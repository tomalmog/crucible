import type { AgentTraceEvent } from "../../types/agent";

interface AgentEventTimelineProps {
  events: AgentTraceEvent[];
}

export function AgentEventTimeline({ events }: AgentEventTimelineProps): React.ReactNode {
  if (events.length === 0) {
    return null;
  }
  return (
    <div className="agent-trace">
      {events.map((event, index) => (
        <div key={index} className={`agent-trace-event agent-trace-event--${event.type}`}>
          <div className="agent-trace-label">{renderLabel(event)}</div>
          <div className="agent-trace-body">{renderBody(event)}</div>
        </div>
      ))}
    </div>
  );
}

function renderLabel(event: AgentTraceEvent): string {
  if (event.type === "assistant_note") return "Plan";
  if (event.type === "tool_call") return `Tool · ${event.tool_name ?? "unknown"}`;
  if (event.type === "tool_result") return `Result · ${event.tool_name ?? "unknown"}`;
  if (event.type === "navigation") return "Navigation";
  if (event.type === "pending_chain") return "Follow-up";
  return "Status";
}

function renderBody(event: AgentTraceEvent): React.ReactNode {
  if (event.type === "tool_call") {
    return event.input_summary || "No arguments";
  }
  if (event.type === "tool_result") {
    return event.output_summary || "No output";
  }
  if (event.type === "navigation") {
    return event.route || "Unknown route";
  }
  if (event.type === "pending_chain") {
    return (
      <>
        {event.job_id && <div>Waiting on {event.job_id}</div>}
        {event.steps?.map((step, index) => <div key={index}>{step}</div>)}
      </>
    );
  }
  return event.text || "Working...";
}
