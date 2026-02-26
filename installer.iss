; ══════════════════════════════════════════════════════════════
; Personal LLM — Windows Installer Script (Inno Setup 6)
;
; Creates a professional installer with:
;   ✅ Install wizard with license/readme
;   ✅ Desktop & Start Menu shortcuts
;   ✅ Creates personal_llm_models folder
;   ✅ Proper uninstaller
;
; Build: "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer.iss
; ══════════════════════════════════════════════════════════════

#define MyAppName "Personal LLM"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "Personal LLM Project"
#define MyAppURL "https://github.com/YOUR_USERNAME/personal-llm"
#define MyAppExeName "PersonalLLM.exe"

[Setup]
; Unique app ID — DO NOT change this between releases
AppId={{A7B3C4D5-E6F7-8901-2345-6789ABCDEF01}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
; Allow user to install without admin rights (per-user install)
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
; Output installer file settings
OutputDir=installer_output
OutputBaseFilename=PersonalLLM_Setup_v{#MyAppVersion}
; Compression — solid LZMA2 for best compression ratio
Compression=lzma2/ultra64
SolidCompression=yes
; Visual settings
WizardStyle=modern
WizardSizePercent=120
; Disk spanning — allow large installs
DiskSpanning=no
; Uninstall settings
UninstallDisplayName={#MyAppName}
; Require Windows 10+
MinVersion=10.0

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Messages]
WelcomeLabel2=This will install [name/ver] on your computer.%n%nPersonal LLM is a fully offline, private AI assistant that runs entirely on your hardware. No internet required after installation.%n%nIMPORTANT: You will need to download AI model files (.gguf) separately after installation. The app will guide you.

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
; Main application files from PyInstaller dist folder
Source: "dist\PersonalLLM\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion
Source: "dist\PersonalLLM\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
; README for reference
Source: "README.md"; DestDir: "{app}"; Flags: ignoreversion
; NOTE: Don't use "skipifsourcedoesntexist" on any file — if it's listed, it must exist

[Dirs]
; Create the models folder so users know where to put their models
Name: "{app}\personal_llm_models"; Permissions: users-modify
; Create the data folder for chat history, knowledge base, etc.
Name: "{app}\personal_llm_data"; Permissions: users-modify
Name: "{app}\personal_llm_data\chat_history"; Permissions: users-modify
Name: "{app}\personal_llm_data\knowledge_db"; Permissions: users-modify
Name: "{app}\personal_llm_data\documents"; Permissions: users-modify

[Icons]
; Start Menu shortcut
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Comment: "Launch Personal LLM — Your Private AI Assistant"
; Start Menu uninstall shortcut
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
; Desktop shortcut (if user chose it)
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon; Comment: "Launch Personal LLM"

[Run]
; Option to launch app after install
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Clean up data files on uninstall (user can choose to keep if they copy them first)
Type: filesandordirs; Name: "{app}\personal_llm_data"

[Code]
// Show a message after installation about downloading models
procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
  begin
    MsgBox(
      'Installation complete!' + #13#10 + #13#10 +
      'IMPORTANT: To use Personal LLM, you need AI model files (.gguf format).' + #13#10 + #13#10 +
      'Place your .gguf model files in:' + #13#10 +
      ExpandConstant('{app}') + '\personal_llm_models\' + #13#10 + #13#10 +
      'Recommended starter model: Phi-3 Mini (~2.4 GB)' + #13#10 +
      'Download from: https://huggingface.co/bartowski/Phi-3.1-mini-4k-instruct-GGUF',
      mbInformation, MB_OK
    );
  end;
end;
