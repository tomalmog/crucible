//! Crucible command execution helpers for Studio.

use crate::commands::crucible_task_store::CommandTaskStore;
use crate::models::{CommandTaskStart, CommandTaskStatus};
use tauri::State;

const ALLOWED_COMMANDS: [&str; 53] = [
    "ingest",
    "filter",
    "train",
    "sft",
    "dpo-train",
    "rlhf-train",
    "lora-train",
    "lora-merge",
    "distill",
    "domain-adapt",
    "distributed-train",
    "export-training",
    "export-spec",
    "chat",
    "run-spec",
    "verify",
    "hardware-profile",
    "benchmark",
    "sweep",
    "compare",
    "replay",
    "model",
    "compute",
    "grpo-train",
    "qlora-train",
    "kto-train",
    "orpo-train",
    "multimodal-train",
    "rlvr-train",
    "suggest",
    "hub",
    "eval",
    "curate",
    "merge",
    "ab-chat",
    "recipe",
    "cloud",
    "synthetic",
    "remote",
    "logit-lens",
    "activation-pca",
    "activation-patch",
    "dispatch",
    "job",
    "onnx-export",
    "safetensors-export",
    "gguf-export",
    "hf-export",
    "agent-chat",
    "run-script",
    "dataset",
    "steer-compute",
    "steer-apply",
];

#[tauri::command]
pub fn start_crucible_command(
    data_root: String,
    args: Vec<String>,
    label: Option<String>,
    config_json: Option<String>,
    task_store: State<'_, CommandTaskStore>,
) -> Result<CommandTaskStart, String> {
    validate_args(&args)?;
    Ok(task_store.start_task(data_root, args, label.unwrap_or_default(), config_json))
}

#[tauri::command]
pub fn get_crucible_command_status(
    task_id: String,
    task_store: State<'_, CommandTaskStore>,
) -> Result<CommandTaskStatus, String> {
    task_store.get_task_status(&task_id)
}

#[tauri::command]
pub fn list_crucible_tasks(
    task_store: State<'_, CommandTaskStore>,
) -> Vec<CommandTaskStatus> {
    task_store.list_all_tasks()
}

#[tauri::command]
pub fn kill_crucible_task(
    task_id: String,
    task_store: State<'_, CommandTaskStore>,
) -> Result<(), String> {
    task_store.kill_task(&task_id)
}

#[tauri::command]
pub fn rename_crucible_task(
    task_id: String,
    label: String,
    task_store: State<'_, CommandTaskStore>,
) -> Result<(), String> {
    task_store.rename_task(&task_id, label)
}

#[tauri::command]
pub fn delete_crucible_task(
    task_id: String,
    task_store: State<'_, CommandTaskStore>,
) -> Result<(), String> {
    task_store.delete_task(&task_id)
}

fn validate_args(args: &[String]) -> Result<(), String> {
    if args.is_empty() {
        return Err("Crucible args must include a command".to_string());
    }
    let command = args[0].as_str();
    if ALLOWED_COMMANDS.contains(&command) {
        Ok(())
    } else {
        Err(format!("Unsupported command '{command}' for Studio execution"))
    }
}

#[cfg(test)]
mod tests {
    use super::validate_args;

    #[test]
    fn validate_args_accepts_supported_command() {
        let args = vec!["train".to_string(), "--epochs".to_string(), "3".to_string()];
        assert!(validate_args(&args).is_ok());
    }

    #[test]
    fn validate_args_accepts_new_commands() {
        for cmd in ["sft", "dpo-train", "rlhf-train", "lora-train", "distill",
                     "domain-adapt",
                     "model", "sweep", "benchmark", "compare", "replay", "compute",
                     "grpo-train", "qlora-train", "kto-train", "orpo-train",
                     "multimodal-train", "rlvr-train", "suggest",
                     "hub", "eval", "curate", "merge", "ab-chat",
                     "recipe", "cloud", "synthetic",
                     "logit-lens", "activation-pca", "activation-patch",
                     "onnx-export",
                     "safetensors-export", "gguf-export",
                     "hf-export"] {
            let args = vec![cmd.to_string()];
            assert!(validate_args(&args).is_ok(), "Expected '{cmd}' to be allowed");
        }
    }

    #[test]
    fn validate_args_rejects_empty_args() {
        let args: Vec<String> = Vec::new();
        assert!(validate_args(&args).is_err());
    }

    #[test]
    fn validate_args_rejects_unsupported_command() {
        let args = vec!["shell".to_string()];
        assert!(validate_args(&args).is_err());
    }
}
