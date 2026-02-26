//! Tauri entrypoint and command wiring for Forge Studio desktop app.

mod commands;
mod models;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(commands::forge_task_store::CommandTaskStore::default())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![
            commands::dataset_queries::get_dataset_dashboard,
            commands::dataset_queries::list_datasets,
            commands::dataset_queries::list_versions,
            commands::dataset_queries::load_training_history,
            commands::dataset_queries::sample_records,
            commands::dataset_queries::version_diff,
            commands::forge_commands::start_forge_command,
            commands::forge_commands::get_forge_command_status,
            commands::forge_commands::list_forge_tasks,
            commands::forge_commands::kill_forge_task,
            commands::forge_commands::rename_forge_task,
            commands::forge_commands::delete_forge_task,
            commands::runtime_queries::list_training_runs,
            commands::runtime_queries::get_lineage_graph,
            commands::runtime_queries::get_hardware_profile,
            commands::config_store::load_training_config,
            commands::config_store::save_training_config
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
