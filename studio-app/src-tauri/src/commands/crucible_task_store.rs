//! Background Crucible command task store and execution worker helpers.

use crate::models::{CommandTaskStart, CommandTaskStatus};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::io::Read;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

const MAX_TASKS: usize = 200;
const MIN_ESTIMATE_SECONDS: u64 = 5;
const MAX_RUNNING_PROGRESS: f64 = 99.0;

/// Commands shown on the Jobs page.
const JOBS_PAGE_COMMANDS: [&str; 33] = [
    "train",
    "sft",
    "dpo-train",
    "rlhf-train",
    "lora-train",
    "lora-merge",
    "distill",
    "domain-adapt",
    "distributed-train",
    "grpo-train",
    "qlora-train",
    "kto-train",
    "orpo-train",
    "multimodal-train",
    "rlvr-train",
    "sweep",
    "logit-lens",
    "activation-pca",
    "activation-patch",
    "linear-probe",
    "sae-train",
    "sae-analyze",
    "eval",
    "dispatch",
    "onnx-export",
    "safetensors-export",
    "gguf-export",
    "hf-export",
    "run-script",
    "steer-compute",
    "steer-apply",
    "hub-download",
    "ingest",
];

#[derive(Clone)]
pub struct CommandTaskStore {
    inner: Arc<CommandTaskStoreInner>,
}

struct CommandTaskStoreInner {
    tasks: Mutex<HashMap<String, TaskRecord>>,
    duration_estimates: Mutex<HashMap<String, f64>>,
    next_task_id: AtomicU64,
}

#[derive(Clone)]
struct TaskRecord {
    task_id: String,
    command: String,
    args: Vec<String>,
    status: TaskLifecycleStatus,
    started_at: Instant,
    finished_at: Option<Instant>,
    estimated_total_seconds: u64,
    stdout: String,
    stderr: String,
    exit_code: Option<i32>,
    pid: Option<u32>,
    label: Option<String>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TaskLifecycleStatus {
    Running,
    Completed,
    Failed,
}

impl Default for CommandTaskStore {
    fn default() -> Self {
        Self {
            inner: Arc::new(CommandTaskStoreInner {
                tasks: Mutex::new(HashMap::new()),
                duration_estimates: Mutex::new(HashMap::new()),
                next_task_id: AtomicU64::new(1),
            }),
        }
    }
}

impl CommandTaskStore {
    pub fn start_task(&self, data_root: String, args: Vec<String>, label: String, config_json: Option<String>) -> CommandTaskStart {
        let command_name = args[0].clone();
        let task_id = self.generate_task_id();
        let estimated_total_seconds = self.estimate_for_command(&command_name);
        let label_opt = if label.is_empty() { None } else { Some(label.clone()) };
        self.insert_running_task(
            task_id.clone(),
            command_name.clone(),
            args.clone(),
            estimated_total_seconds,
            label_opt,
        );

        // Write the job record synchronously before spawning the thread so the
        // file exists before start_crucible_command returns to the UI.
        // Dispatch is excluded: Python writes the real job record itself.
        if JOBS_PAGE_COMMANDS.contains(&command_name.as_str()) && command_name != "dispatch" {
            write_job_record(&data_root, &task_id, &command_name, "running", "", "", &label, &config_json);
        }

        let task_store = self.clone();
        let task_id_for_thread = task_id.clone();
        std::thread::spawn(move || {
            task_store.execute_task(task_id_for_thread, data_root, command_name, args, label, config_json);
        });

        CommandTaskStart {
            task_id,
            estimated_total_seconds,
        }
    }

    pub fn get_task_status(&self, task_id: &str) -> Result<CommandTaskStatus, String> {
        let task = {
            let tasks = self
                .inner
                .tasks
                .lock()
                .map_err(|_| "Task store lock poisoned".to_string())?;
            tasks
                .get(task_id)
                .cloned()
                .ok_or_else(|| format!("Unknown task id '{task_id}'"))?
        };
        Ok(task_to_status(task))
    }

    pub fn list_all_tasks(&self) -> Vec<CommandTaskStatus> {
        let tasks = match self.inner.tasks.lock() {
            Ok(guard) => guard,
            Err(_) => return Vec::new(),
        };
        let mut result: Vec<CommandTaskStatus> = tasks
            .values()
            .filter(|t| JOBS_PAGE_COMMANDS.contains(&t.command.as_str()))
            .cloned()
            .map(task_to_status)
            .collect();
        result.sort_by(|a, b| b.task_id.cmp(&a.task_id));
        result
    }

    pub fn rename_task(&self, task_id: &str, label: String) -> Result<(), String> {
        let mut tasks = self
            .inner
            .tasks
            .lock()
            .map_err(|_| "Task store lock poisoned".to_string())?;
        let task = tasks
            .get_mut(task_id)
            .ok_or_else(|| format!("Unknown task id '{task_id}'"))?;
        task.label = if label.is_empty() { None } else { Some(label) };
        Ok(())
    }

    pub fn delete_task(&self, task_id: &str) -> Result<(), String> {
        let mut tasks = self
            .inner
            .tasks
            .lock()
            .map_err(|_| "Task store lock poisoned".to_string())?;
        let task = tasks
            .get(task_id)
            .ok_or_else(|| format!("Unknown task id '{task_id}'"))?;
        if task.status == TaskLifecycleStatus::Running {
            return Err("Cannot delete a running task — kill it first".to_string());
        }
        tasks.remove(task_id);
        Ok(())
    }

    /// Kill all running child processes. Called on app exit to prevent orphans.
    pub fn kill_all_running(&self) {
        let pids: Vec<u32> = {
            let tasks = match self.inner.tasks.lock() {
                Ok(guard) => guard,
                Err(_) => return,
            };
            tasks
                .values()
                .filter(|t| t.status == TaskLifecycleStatus::Running)
                .filter_map(|t| t.pid)
                .collect()
        };
        for pid in pids {
            unsafe {
                libc::kill(pid as i32, libc::SIGTERM);
            }
        }
    }

    pub fn kill_task(&self, task_id: &str) -> Result<(), String> {
        let pid = {
            let tasks = self
                .inner
                .tasks
                .lock()
                .map_err(|_| "Task store lock poisoned".to_string())?;
            let task = tasks
                .get(task_id)
                .ok_or_else(|| format!("Unknown task id '{task_id}'"))?;
            if task.status != TaskLifecycleStatus::Running {
                return Err(format!("Task '{task_id}' is not running"));
            }
            task.pid.ok_or_else(|| format!("Task '{task_id}' has no PID"))?
        };
        let ret = unsafe { libc::kill(pid as i32, libc::SIGTERM) };
        if ret != 0 {
            return Err(format!("Failed to kill process {pid}"));
        }
        Ok(())
    }

    fn execute_task(&self, task_id: String, data_root: String, command_name: String, args: Vec<String>, _label: String, config_json: Option<String>) {
        // Write payload file for agent-chat so the CLI can read it
        let mut args = args;
        if command_name == "agent-chat" {
            if let Some(ref json) = config_json {
                let payload_dir = resolve_data_root(&data_root).join("agent");
                let _ = fs::create_dir_all(&payload_dir);
                let payload_path = payload_dir.join("_payload.json");
                let _ = fs::write(&payload_path, json);
                // Replace --payload-file value with the actual written path
                args = vec![
                    "agent-chat".to_string(),
                    "--payload-file".to_string(),
                    payload_path.to_string_lossy().to_string(),
                ];
            }
        }

        let working_directory = workspace_root_dir();
        let crucible_bin = resolve_crucible_binary(&working_directory);
        let spawn_result = Command::new(crucible_bin)
            .current_dir(working_directory)
            .env("PYTHONUNBUFFERED", "1")
            .env("CRUCIBLE_TASK_ID", &task_id)
            .arg("--data-root")
            .arg(&data_root)
            .args(args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn();

        match spawn_result {
            Ok(mut child) => {
                let child_pid = child.id();
                if let Ok(mut tasks) = self.inner.tasks.lock() {
                    if let Some(task) = tasks.get_mut(&task_id) {
                        task.pid = Some(child_pid);
                    }
                }
                self.stream_child_output(&task_id, &mut child);
                self.finalize_child(&task_id, &command_name, &data_root, &mut child);
            }
            Err(error) => {
                self.fail_task(&task_id, &command_name, &data_root, error.to_string());
            }
        }
    }

    fn stream_child_output(&self, task_id: &str, child: &mut std::process::Child) {
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();

        // Drain stderr on a separate thread to prevent deadlock.
        // If stderr's OS pipe buffer fills up while we only read stdout,
        // the child blocks on stderr writes and never closes stdout.
        let stderr_store = self.clone();
        let stderr_task_id = task_id.to_string();
        let stderr_thread = stderr.map(|mut stderr_pipe| {
            std::thread::spawn(move || {
                let mut buf = [0u8; 4096];
                loop {
                    match stderr_pipe.read(&mut buf) {
                        Ok(0) | Err(_) => break,
                        Ok(n) => {
                            let chunk = String::from_utf8_lossy(&buf[..n]).to_string();
                            if let Ok(mut tasks) = stderr_store.inner.tasks.lock() {
                                if let Some(task) = tasks.get_mut(&stderr_task_id) {
                                    task.stderr.push_str(&chunk);
                                }
                            }
                        }
                    }
                }
            })
        });

        // Read stdout on the current thread.
        if let Some(mut stdout_pipe) = stdout {
            let mut buf = [0u8; 4096];
            loop {
                match stdout_pipe.read(&mut buf) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => {
                        let chunk = String::from_utf8_lossy(&buf[..n]).to_string();
                        if let Ok(mut tasks) = self.inner.tasks.lock() {
                            if let Some(task) = tasks.get_mut(task_id) {
                                task.stdout.push_str(&chunk);
                            }
                        }
                    }
                }
            }
        }

        // Wait for stderr drain to finish.
        if let Some(handle) = stderr_thread {
            let _ = handle.join();
        }
    }

    fn finalize_child(&self, task_id: &str, command_name: &str, data_root: &str, child: &mut std::process::Child) {
        let exit_status = child.wait();

        let now = Instant::now();
        let mut observed_elapsed_seconds = None;
        let mut final_state = "failed";
        let mut stdout_snapshot = String::new();
        let mut stderr_snapshot = String::new();
        let mut label_snapshot = String::new();
        if let Ok(mut tasks) = self.inner.tasks.lock() {
            if let Some(task) = tasks.get_mut(task_id) {
                let exit_code = exit_status
                    .map(|s| s.code().unwrap_or(-1))
                    .unwrap_or(-1);
                task.exit_code = Some(exit_code);
                task.status = if exit_code == 0 {
                    TaskLifecycleStatus::Completed
                } else {
                    TaskLifecycleStatus::Failed
                };
                task.finished_at = Some(now);
                observed_elapsed_seconds = Some(task.started_at.elapsed().as_secs_f64().max(1.0));
                final_state = if exit_code == 0 { "completed" } else { "failed" };
                stdout_snapshot = task.stdout.clone();
                stderr_snapshot = task.stderr.clone();
                label_snapshot = task.label.clone().unwrap_or_default();
            }
        }
        if let Some(observed_seconds) = observed_elapsed_seconds {
            self.update_duration_estimate(command_name, observed_seconds);
        }
        if JOBS_PAGE_COMMANDS.contains(&command_name) {
            if command_name == "dispatch" {
                // Dispatch: Python creates real job-XXXX records. We don't write a pending
                // record up front, so on success there's nothing to clean up.
                // On failure, Python never created a job record — write a failed record now.
                if final_state != "completed" {
                    let err_msg = {
                        let len = stderr_snapshot.len();
                        if len > 2000 { stderr_snapshot[len-2000..].to_string() } else { stderr_snapshot.clone() }
                    };
                    write_job_record(data_root, task_id, "dispatch", "failed", "", &err_msg, &label_snapshot, &None);
                    update_job_record(data_root, task_id, "failed", "", &err_msg, &stdout_snapshot, &stderr_snapshot);
                }
            } else {
                let model_path = extract_model_path(&stdout_snapshot);
                let err_msg = if final_state == "failed" {
                    let len = stderr_snapshot.len();
                    if len > 2000 { stderr_snapshot[len-2000..].to_string() } else { stderr_snapshot.clone() }
                } else { String::new() };
                update_job_record(data_root, task_id, final_state, &model_path, &err_msg, &stdout_snapshot, &stderr_snapshot);
            }
        }
    }

    fn fail_task(&self, task_id: &str, command_name: &str, data_root: &str, error_message: String) {
        let now = Instant::now();
        let mut observed_elapsed_seconds = None;
        if let Ok(mut tasks) = self.inner.tasks.lock() {
            if let Some(task) = tasks.get_mut(task_id) {
                task.exit_code = Some(-1);
                task.status = TaskLifecycleStatus::Failed;
                task.stderr = format!("Failed to run crucible command: {error_message}");
                task.finished_at = Some(now);
                observed_elapsed_seconds = Some(task.started_at.elapsed().as_secs_f64().max(1.0));
            }
        }
        if let Some(observed_seconds) = observed_elapsed_seconds {
            self.update_duration_estimate(command_name, observed_seconds);
        }
        // Update unified JobRecord — dispatch failures need to be visible too
        if JOBS_PAGE_COMMANDS.contains(&command_name) {
            update_job_record(data_root, task_id, "failed", "", &error_message, "", &error_message);
        }
    }

    fn generate_task_id(&self) -> String {
        let value = self.inner.next_task_id.fetch_add(1, Ordering::Relaxed);
        format!("crucible-task-{value}")
    }

    fn insert_running_task(
        &self,
        task_id: String,
        command_name: String,
        args: Vec<String>,
        estimated_total_seconds: u64,
        label: Option<String>,
    ) {
        if let Ok(mut tasks) = self.inner.tasks.lock() {
            tasks.insert(
                task_id.clone(),
                TaskRecord {
                    task_id,
                    command: command_name,
                    args,
                    status: TaskLifecycleStatus::Running,
                    started_at: Instant::now(),
                    finished_at: None,
                    estimated_total_seconds,
                    stdout: String::new(),
                    stderr: String::new(),
                    exit_code: None,
                    pid: None,
                    label,
                },
            );
            prune_finished_tasks(&mut tasks);
        }
    }

    fn estimate_for_command(&self, command_name: &str) -> u64 {
        let default_seconds = default_estimate_seconds(command_name);
        let guard = self.inner.duration_estimates.lock();
        if let Ok(estimates) = guard {
            if let Some(average_seconds) = estimates.get(command_name) {
                return average_seconds.round().max(MIN_ESTIMATE_SECONDS as f64) as u64;
            }
        }
        default_seconds
    }

    fn update_duration_estimate(&self, command_name: &str, observed_seconds: f64) {
        if let Ok(mut estimates) = self.inner.duration_estimates.lock() {
            let next_average = if let Some(current_average) = estimates.get(command_name).copied() {
                current_average * 0.7 + observed_seconds * 0.3
            } else {
                observed_seconds
            };
            estimates.insert(command_name.to_string(), next_average);
        }
    }
}

fn prune_finished_tasks(tasks: &mut HashMap<String, TaskRecord>) {
    if tasks.len() <= MAX_TASKS {
        return;
    }
    let mut removable: Vec<String> = tasks
        .iter()
        .filter(|(_, task)| task.status != TaskLifecycleStatus::Running)
        .map(|(task_id, _)| task_id.clone())
        .collect();
    removable.sort();
    let excess = tasks.len().saturating_sub(MAX_TASKS);
    for task_id in removable.into_iter().take(excess) {
        tasks.remove(&task_id);
    }
}

fn task_to_status(task: TaskRecord) -> CommandTaskStatus {
    let elapsed_seconds = match task.finished_at {
        Some(finished) => finished.duration_since(task.started_at).as_secs(),
        None => task.started_at.elapsed().as_secs(),
    };
    let status = task_status_name(task.status).to_string();
    let remaining_seconds = if task.status == TaskLifecycleStatus::Running {
        task.estimated_total_seconds.saturating_sub(elapsed_seconds)
    } else {
        0
    };
    let progress_percent = match task.status {
        TaskLifecycleStatus::Running => {
            running_progress_percent(elapsed_seconds, task.estimated_total_seconds)
        }
        TaskLifecycleStatus::Completed | TaskLifecycleStatus::Failed => 100.0,
    };
    CommandTaskStatus {
        task_id: task.task_id,
        status,
        command: task.command,
        args: task.args,
        exit_code: task.exit_code,
        stdout: task.stdout,
        stderr: task.stderr,
        elapsed_seconds,
        estimated_total_seconds: task.estimated_total_seconds,
        remaining_seconds,
        progress_percent,
        label: task.label,
    }
}

fn running_progress_percent(elapsed_seconds: u64, estimated_total_seconds: u64) -> f64 {
    let estimate = estimated_total_seconds.max(MIN_ESTIMATE_SECONDS);
    let raw = (elapsed_seconds as f64 / estimate as f64) * 100.0;
    raw.clamp(1.0, MAX_RUNNING_PROGRESS)
}

fn task_status_name(status: TaskLifecycleStatus) -> &'static str {
    match status {
        TaskLifecycleStatus::Running => "running",
        TaskLifecycleStatus::Completed => "completed",
        TaskLifecycleStatus::Failed => "failed",
    }
}

fn default_estimate_seconds(command_name: &str) -> u64 {
    match command_name {
        "ingest" => 60,
        "filter" => 30,
        "train" | "sft" | "dpo-train" | "rlhf-train" | "lora-train"
        | "qlora-train" | "grpo-train" | "kto-train" | "orpo-train"
        | "multimodal-train" | "rlvr-train" | "distill" | "domain-adapt" => 240,
        "sweep" => 900,
        "eval" => 300,
        "export-training" => 60,
        "hub-download" => 120,
        "versions" => 8,
        "chat" => 20,
        _ => 30,
    }
}

/// Resolve a possibly-relative data_root to an absolute path anchored at the workspace root.
fn resolve_data_root(data_root: &str) -> PathBuf {
    let raw = Path::new(data_root);
    if raw.is_absolute() { raw.to_path_buf() } else { workspace_root_dir().join(raw) }
}

/// Write a unified JobRecord JSON to .crucible/jobs/ so the unified UI can see it.
fn write_job_record(
    data_root: &str,
    task_id: &str,
    command: &str,
    state: &str,
    model_path: &str,
    error_message: &str,
    label: &str,
    config_json: &Option<String>,
) {
    let jobs_dir = resolve_data_root(data_root).join("jobs");
    if fs::create_dir_all(&jobs_dir).is_err() {
        return;
    }
    let now = utc_iso_now();
    let config_value: serde_json::Value = config_json
        .as_deref()
        .and_then(|s| serde_json::from_str(s).ok())
        .unwrap_or_else(|| serde_json::json!({}));
    let record = serde_json::json!({
        "job_id": task_id,
        "backend": "local",
        "job_type": command,
        "state": state,
        "created_at": now,
        "updated_at": now,
        "label": label,
        "backend_job_id": "",
        "backend_cluster": "",
        "backend_output_dir": "",
        "backend_log_path": "",
        "model_path": model_path,
        "model_path_local": model_path,
        "model_name": "",
        "error_message": error_message,
        "progress_percent": if state == "running" { 0.0 } else { 100.0 },
        "submit_phase": "",
        "is_sweep": command == "sweep",
        "sweep_trial_count": 0,
        "config": config_value
    });
    let path = jobs_dir.join(format!("{task_id}.json"));
    let _ = fs::write(&path, serde_json::to_string_pretty(&record).unwrap_or_default());
}

/// Update just the state, model_path, error_message, and stdout/stderr of an existing job record.
fn update_job_record(
    data_root: &str,
    task_id: &str,
    state: &str,
    model_path: &str,
    error_message: &str,
    stdout: &str,
    stderr: &str,
) {
    let path = resolve_data_root(data_root).join("jobs").join(format!("{task_id}.json"));
    if !path.exists() {
        return;
    }
    let contents = match fs::read_to_string(&path) {
        Ok(c) => c,
        Err(_) => return,
    };
    let mut record: serde_json::Value = match serde_json::from_str(&contents) {
        Ok(v) => v,
        Err(_) => return,
    };
    if let Some(obj) = record.as_object_mut() {
        obj.insert("state".to_string(), serde_json::json!(state));
        obj.insert("updated_at".to_string(), serde_json::json!(utc_iso_now()));
        obj.insert("progress_percent".to_string(), serde_json::json!(100.0));
        // Clear submit_phase on terminal states so stale spinner text isn't shown
        if state == "completed" || state == "failed" || state == "cancelled" {
            obj.insert("submit_phase".to_string(), serde_json::json!(""));
        }
        if !model_path.is_empty() {
            obj.insert("model_path".to_string(), serde_json::json!(model_path));
            obj.insert("model_path_local".to_string(), serde_json::json!(model_path));
        }
        if !error_message.is_empty() {
            obj.insert("error_message".to_string(), serde_json::json!(error_message));
        }
        if !stdout.is_empty() {
            obj.insert("stdout".to_string(), serde_json::json!(stdout));
        }
        if !stderr.is_empty() {
            obj.insert("stderr".to_string(), serde_json::json!(stderr));
        }
    }
    let _ = fs::write(&path, serde_json::to_string_pretty(&record).unwrap_or_default());
}

/// Extract model_path=... from stdout — uses the LAST occurrence, since some
/// commands print intermediate paths before the final registered model path.
fn extract_model_path(stdout: &str) -> String {
    let mut last = String::new();
    for line in stdout.lines() {
        if let Some(rest) = line.strip_prefix("model_path=") {
            last = rest.trim().to_string();
        }
    }
    last
}

/// UTC ISO-8601 timestamp without chrono dependency.
fn utc_iso_now() -> String {
    let dur = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let total_secs = dur.as_secs();
    let secs_in_day: u64 = 86400;
    let mut days = total_secs / secs_in_day;
    let day_secs = total_secs % secs_in_day;
    let h = day_secs / 3600;
    let m = (day_secs % 3600) / 60;
    let s = day_secs % 60;
    // Simple days-since-epoch to Y-M-D (good enough for sorting)
    let mut year: u64 = 1970;
    loop {
        let days_in_year = if year % 4 == 0 && (year % 100 != 0 || year % 400 == 0) { 366 } else { 365 };
        if days < days_in_year { break; }
        days -= days_in_year;
        year += 1;
    }
    let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let month_days: [u64; 12] = [31, if leap { 29 } else { 28 }, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let mut month: u64 = 1;
    for md in &month_days {
        if days < *md { break; }
        days -= md;
        month += 1;
    }
    let day = days + 1;
    format!("{year:04}-{month:02}-{day:02}T{h:02}:{m:02}:{s:02}Z")
}

pub fn workspace_root_dir() -> PathBuf {
    // `CARGO_MANIFEST_DIR` points to `studio-app/src-tauri`.
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

pub fn resolve_crucible_binary(workspace_root: &Path) -> PathBuf {
    // Try both names — the binary may be called "crucible" or "forge"
    for name in ["crucible", "forge"] {
        let venv_binary = workspace_root.join(format!(".venv/bin/{name}"));
        if venv_binary.exists() {
            return venv_binary;
        }
    }
    PathBuf::from("crucible")
}

#[cfg(test)]
mod tests {
    use super::{default_estimate_seconds, running_progress_percent};

    #[test]
    fn running_progress_is_bounded_before_completion() {
        let progress = running_progress_percent(1, 600);
        assert!(progress >= 1.0 && progress < 100.0);
    }

    #[test]
    fn default_estimate_returns_expected_values() {
        assert_eq!(default_estimate_seconds("train"), 240);
        assert_eq!(default_estimate_seconds("versions"), 8);
        assert_eq!(default_estimate_seconds("chat"), 20);
        assert_eq!(default_estimate_seconds("unknown"), 30);
    }
}
