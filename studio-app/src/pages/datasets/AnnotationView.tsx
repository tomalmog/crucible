import { useState } from "react";

interface AnnotationTask {
  id: string;
  prompt: string;
  responseA: string;
  responseB: string;
  label: string;
}

export function AnnotationView() {
  const [tasks, setTasks] = useState<AnnotationTask[]>([
    { id: "1", prompt: "Example prompt", responseA: "Response A", responseB: "Response B", label: "" },
  ]);

  function setLabel(index: number, label: string) {
    const updated = [...tasks];
    updated[index] = { ...updated[index], label };
    setTasks(updated);
  }

  const completed = tasks.filter((t) => t.label).length;

  return (
    <div className="panel stack-md">
      <h3>Data Annotation</h3>
      <p className="text-muted">Completed: {completed} / {tasks.length}</p>
      {tasks.map((task, i) => (
        <div key={task.id} className="stack-sm section-divider">
          <p><strong>Prompt:</strong> {task.prompt}</p>
          <div className="grid-2">
            <div className="panel"><h4>Response A</h4><p>{task.responseA}</p></div>
            <div className="panel"><h4>Response B</h4><p>{task.responseB}</p></div>
          </div>
          <div className="row">
            <button className={`btn btn-sm${task.label === "a" ? " btn-primary" : ""}`} onClick={() => setLabel(i, "a")}>A is Better</button>
            <button className={`btn btn-sm${task.label === "tie" ? " btn-primary" : ""}`} onClick={() => setLabel(i, "tie")}>Tie</button>
            <button className={`btn btn-sm${task.label === "b" ? " btn-primary" : ""}`} onClick={() => setLabel(i, "b")}>B is Better</button>
          </div>
        </div>
      ))}
    </div>
  );
}
