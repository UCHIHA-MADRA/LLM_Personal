use std::process::Command;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
  tauri::Builder::default()
    .plugin(tauri_plugin_shell::init())
    .setup(|app| {
      #[cfg(desktop)]
      {
          // Spawn the Python FastAPI Backend as a headless sidecar process
          let _child = Command::new("python")
            .current_dir("..")
            .arg("personal_llm/api.py")
            .spawn()
            .expect("Failed to start python backend");
            
          println!("🚀 Tauri Desktop Shell started python backend thread.");
      }
      
      if cfg!(debug_assertions) {
        app.handle().plugin(
          tauri_plugin_log::Builder::default()
            .level(log::LevelFilter::Info)
            .build(),
        )?;
      }
      Ok(())
    })
    .run(tauri::generate_context!())
    .expect("error while running tauri application");
}
