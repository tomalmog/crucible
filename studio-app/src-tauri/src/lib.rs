//! Tauri entrypoint and command wiring for Crucible Studio desktop app.

mod commands;
mod models;

use tauri::{Manager, RunEvent};

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let app = tauri::Builder::default()
        .manage(commands::crucible_task_store::CommandTaskStore::default())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![
            commands::dataset_queries::get_dataset_dashboard,
            commands::dataset_queries::delete_dataset,
            commands::dataset_queries::list_datasets,
            commands::dataset_queries::load_training_history,
            commands::dataset_queries::sample_records,
            commands::crucible_commands::start_crucible_command,
            commands::crucible_commands::get_crucible_command_status,
            commands::crucible_commands::list_crucible_tasks,
            commands::crucible_commands::kill_crucible_task,
            commands::crucible_commands::rename_crucible_task,
            commands::crucible_commands::delete_crucible_task,
            commands::runtime_queries::list_training_runs,
            commands::runtime_queries::get_lineage_graph,
            commands::runtime_queries::get_hardware_profile,
            commands::model_queries::list_models,
            commands::model_queries::get_model_architecture,
            commands::model_queries::get_model_index_mtime,
            commands::config_store::load_training_config,
            commands::config_store::save_training_config,
            commands::config_store::write_text_file,
            commands::remote_queries::list_clusters,
            commands::remote_queries::list_remote_jobs,
            commands::remote_queries::get_remote_job,
            commands::remote_queries::delete_remote_job,
            commands::remote_cli_ops::get_remote_job_result,
            commands::remote_cli_ops::get_remote_job_logs,
            commands::remote_cli_ops::sync_remote_job_status,
            commands::remote_cli_ops::cancel_remote_job,
            commands::remote_cli_ops::list_remote_datasets,
            commands::remote_cli_ops::push_dataset_to_cluster,
            commands::remote_cli_ops::pull_dataset_from_cluster,
            commands::remote_cli_ops::delete_remote_dataset_cmd,
            commands::remote_cli_ops::get_remote_model_sizes,
            commands::remote_cli_ops::get_cluster_info,
            commands::job_queries::list_unified_jobs,
            commands::job_queries::get_unified_job,
            commands::job_queries::delete_unified_job,
            commands::job_queries::sync_unified_job_state,
            commands::job_queries::cancel_unified_job,
            commands::job_queries::get_unified_job_logs,
            commands::job_queries::get_unified_job_result,
            commands::storage_queries::get_storage_breakdown,
            commands::storage_queries::list_orphaned_runs,
            commands::storage_queries::delete_orphaned_runs,
            commands::storage_queries::clear_cache
        ])
        .build(tauri::generate_context!())
        .expect("error while building tauri application");

    app.run(|app_handle, event| {
        if let RunEvent::ExitRequested { .. } = event {
            let store = app_handle.state::<commands::crucible_task_store::CommandTaskStore>();
            store.kill_all_running();
        }
    });
}
