# Privacy Policy — Personal LLM

**Last Updated:** March 27, 2026  
**Version:** 2.0.2

---

## Our Privacy Promise

Personal LLM is built on a single, uncompromising principle: **your data never leaves your machine.** Unlike cloud-based AI services, Personal LLM runs entirely on your local hardware. We do not collect, transmit, store, or process any user data on external servers.

---

## 1. Data Collection

**We collect nothing.** Specifically:

| Data Type | Collected? | Details |
|---|---|---|
| Conversations | ❌ No | Stored only on your local filesystem |
| Uploaded documents | ❌ No | Stored only in your local `documents/` directory |
| API keys | ❌ No | Stored only in your local `settings.json` |
| Usage analytics | ❌ No | No telemetry of any kind |
| Crash reports | ❌ No | Errors are logged locally only |
| IP addresses | ❌ No | No network activity except user-initiated |
| Device information | ❌ No | Hardware detection is local-only |

---

## 2. Data Storage

All data is stored locally on your machine in the following locations:

- **Development mode:** Inside the `LLM_Personal/personal_llm/` project directory
- **Desktop app (.exe):** In `%LOCALAPPDATA%\PersonalLLM` (Windows)
- **macOS:** In `~/Library/Application Support/PersonalLLM`
- **Linux:** In `~/.local/share/personal-llm`

### What is stored locally:

- **Conversations:** JSON files in `chat_history/`
- **Knowledge Base:** ChromaDB database in `knowledge_db/`
- **Uploaded Documents:** Original files in `documents/`
- **Settings:** API keys and preferences in `settings.json`
- **Logs:** Application logs in `logs/` (auto-rotated, 7-day retention)
- **Models:** Downloaded `.gguf` model files in `personal_llm_models/`

---

## 3. Network Activity

Personal LLM makes **zero outbound network requests** by default. The only exceptions are:

1. **Model Downloads** — When you explicitly choose to download a model, a request is made to `huggingface.co` to fetch the model file. This is a one-time action per model.

2. **Cloud AI (Optional)** — If you configure API keys for Gemini or Claude and explicitly choose to use cloud chat, your messages for that specific conversation are sent to the respective provider (Google or Anthropic). **This is entirely opt-in** and disabled by default.

3. **No Auto-Updates** — Personal LLM does not phone home, check for updates, or download anything without your explicit action. This is by design.

---

## 4. Data Deletion

You have complete control over your data:

- **Individual conversations:** Delete via the UI
- **Individual documents:** Remove via the Knowledge Base manager
- **Complete data wipe:** Use Settings → Privacy → "Wipe All Data" (requires typing `DELETE ALL MY DATA` as confirmation)
- **Manual deletion:** Simply delete the data directories listed in Section 2

After deletion, data is removed from disk. There are no cloud backups, no "soft deletes," and no way for us to recover your data — because we never had it.

---

## 5. Third-Party Services

| Service | When Used | Data Shared |
|---|---|---|
| HuggingFace Hub | Model downloads (user-initiated) | Download request only (no user data) |
| Google Gemini API | Cloud chat (opt-in, requires API key) | Conversation messages for that session |
| Anthropic Claude API | Cloud chat (opt-in, requires API key) | Conversation messages for that session |

**No other third-party services are contacted.** There are no ads, no tracking pixels, no analytics scripts, and no CDN requests.

---

## 6. Security

- **CORS Protection:** The API server restricts cross-origin requests to localhost and the local network only
- **No Authentication Bypass:** Destructive operations require explicit confirmation
- **Local-Only Binding:** When running as a desktop app, the server binds to `127.0.0.1` only
- **No Telemetry:** Zero outbound telemetry, analytics, or crash reporting

---

## 7. Children's Privacy

Personal LLM does not collect any data from anyone, including children. The application runs entirely offline and does not have user accounts.

---

## 8. Changes to This Policy

This privacy policy may be updated with new versions of Personal LLM. Changes will be documented in the project's changelog. Since we collect no data, policy changes will primarily reflect new features or third-party integrations.

---

## 9. Contact

For privacy questions or concerns, please open an issue on the [GitHub repository](https://github.com/UCHIHA-MADRA/LLM_Personal/issues).

---

**Summary:** Personal LLM is a local-first application. Your conversations, documents, API keys, and all other data remain on your machine. We cannot see, access, or sell your data because we never receive it.
