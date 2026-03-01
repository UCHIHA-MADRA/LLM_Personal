import React, { useState, useRef, useEffect } from 'react';
import {
  StyleSheet,
  View,
  Text,
  TextInput,
  TouchableOpacity,
  FlatList,
  KeyboardAvoidingView,
  Platform,
  StatusBar,
  ActivityIndicator,
  Modal,
  ScrollView,
  Alert,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { SafeAreaProvider, useSafeAreaInsets } from 'react-native-safe-area-context';
import { initLlama, type LlamaContext } from 'llama.rn';
import * as FileSystem from 'expo-file-system';

// ─── CONFIG ───
const PC_LAN_IP = '192.168.0.102';
const getApiBase = (ip: string) => `http://${ip}:8000`;

// ─── On-Device Models (small enough for phone RAM) ───
const DEVICE_MODELS = [
  {
    id: 'smollm2-360m',
    name: 'SmolLM2 360M',
    description: 'Ultra-light model, very fast. Good for simple tasks.',
    size_mb: 229,
    url: 'https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct-GGUF/resolve/main/smollm2-360m-instruct-q4_k_m.gguf',
    filename: 'smollm2-360m-instruct-q4_k_m.gguf',
  },
  {
    id: 'smollm2-1.7b',
    name: 'SmolLM2 1.7B',
    description: 'Good balance of quality and speed for phones.',
    size_mb: 1060,
    url: 'https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF/resolve/main/smollm2-1.7b-instruct-q4_k_m.gguf',
    filename: 'smollm2-1.7b-instruct-q4_k_m.gguf',
  },
  {
    id: 'qwen2.5-1.5b',
    name: 'Qwen 2.5 1.5B',
    description: 'High quality small model by Alibaba.',
    size_mb: 1060,
    url: 'https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf',
    filename: 'qwen2.5-1.5b-instruct-q4_k_m.gguf',
  },
  {
    id: 'phi3-mini-3.8b',
    name: 'Phi-3 Mini 3.8B',
    description: 'Microsoft\'s best small model. Needs 4GB+ free RAM.',
    size_mb: 2390,
    url: 'https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf',
    filename: 'Phi-3-mini-4k-instruct-q4.gguf',
  },
];

// ─── Cloud Model Configs (work directly from the phone) ───
const CLOUD_MODELS = [
  { id: 'groq-llama3-8b', name: 'Llama 3 8B (Groq)', provider: 'groq', model: 'llama3-8b-8192', free: true },
  { id: 'groq-llama3-70b', name: 'Llama 3 70B (Groq)', provider: 'groq', model: 'llama3-70b-8192', free: true },
  { id: 'groq-mixtral', name: 'Mixtral 8x7B (Groq)', provider: 'groq', model: 'mixtral-8x7b-32768', free: true },
  { id: 'groq-gemma2', name: 'Gemma 2 9B (Groq)', provider: 'groq', model: 'gemma2-9b-it', free: true },
  { id: 'openai-gpt4o-mini', name: 'GPT-4o Mini (OpenAI)', provider: 'openai', model: 'gpt-4o-mini', free: false },
];

const PROVIDER_ENDPOINTS: Record<string, string> = {
  groq: 'https://api.groq.com/openai/v1/chat/completions',
  openai: 'https://api.openai.com/v1/chat/completions',
};

type Role = 'user' | 'assistant';
type Message = { id: string; role: Role; content: string };
type ModelStatus = { loaded: boolean; name?: string; size_gb?: number };
type ChatMode = 'device' | 'cloud' | 'local';

function AppContent() {
  const insets = useSafeAreaInsets();

  const [messages, setMessages] = useState<Message[]>([
    { id: '1', role: 'assistant', content: 'Hello! I am your Personal LLM. Ask me anything!' },
  ]);
  const [input, setInput] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [status, setStatus] = useState<ModelStatus>({ loaded: false });
  const [catalog, setCatalog] = useState<any[]>([]);
  const [isModalVisible, setModalVisible] = useState(false);
  const [isLoadingModel, setIsLoadingModel] = useState(false);

  // Chat mode
  const [chatMode, setChatMode] = useState<ChatMode>('device');
  const [selectedCloudModel, setSelectedCloudModel] = useState(CLOUD_MODELS[0]);

  // On-device state
  const [llamaContext, setLlamaContext] = useState<LlamaContext | null>(null);
  const [deviceModelName, setDeviceModelName] = useState('');
  const [isDownloadingDevice, setIsDownloadingDevice] = useState(false);
  const [deviceDownloadProgress, setDeviceDownloadProgress] = useState(0);
  const [isLoadingDevice, setIsLoadingDevice] = useState(false);
  const [downloadedModels, setDownloadedModels] = useState<string[]>([]);

  // PC connection state
  const [isDownloading, setIsDownloading] = useState(false);
  const [downloadProgress, setDownloadProgress] = useState(0);

  // UI States
  const [isSidebarOpen, setSidebarOpen] = useState(false);
  const [isSettingsOpen, setSettingsOpen] = useState(false);
  const [conversations, setConversations] = useState<any[]>([]);
  const [currentConvId, setCurrentConvId] = useState<string | null>(null);

  // Settings
  const [settings, setSettings] = useState({ openai_key: '', groq_key: '', together_key: '' });
  const [isSavingSettings, setIsSavingSettings] = useState(false);
  const [backendIp, setBackendIp] = useState(PC_LAN_IP);
  const [isConnected, setIsConnected] = useState(false);
  const API_BASE = getApiBase(backendIp);

  const flatListRef = useRef<FlatList>(null);
  const modelsDir = `${FileSystem.documentDirectory}models/`;

  // Hermes-compatible timeout fetch
  const fetchWithTimeout = (url: string, ms = 3000): Promise<Response> => {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), ms);
    return fetch(url, { signal: controller.signal }).finally(() => clearTimeout(timer));
  };

  // ── Check which device models are already downloaded ──
  const checkDownloadedModels = async () => {
    try {
      const dirInfo = await FileSystem.getInfoAsync(modelsDir);
      if (!dirInfo.exists) {
        await FileSystem.makeDirectoryAsync(modelsDir, { intermediates: true });
        setDownloadedModels([]);
        return;
      }
      const files = await FileSystem.readDirectoryAsync(modelsDir);
      setDownloadedModels(files.filter(f => f.endsWith('.gguf')));
    } catch {
      setDownloadedModels([]);
    }
  };

  // ── Backend connectivity ──
  const fetchStatus = () => {
    fetchWithTimeout(`${API_BASE}/api/status`)
      .then(res => res.json())
      .then(data => { setStatus(data); setIsConnected(true); })
      .catch(() => { setStatus({ loaded: false }); setIsConnected(false); });
  };

  const fetchCatalog = () => {
    fetchWithTimeout(`${API_BASE}/api/models`)
      .then(res => res.json())
      .then(data => setCatalog(data.catalog || []))
      .catch(() => { });
  };

  const fetchConversations = () => {
    fetchWithTimeout(`${API_BASE}/api/conversations`)
      .then(res => res.json())
      .then(data => setConversations(data || []))
      .catch(() => { });
  };

  const fetchSettings = () => {
    fetchWithTimeout(`${API_BASE}/api/settings`)
      .then(res => res.json())
      .then(data => setSettings({
        openai_key: data.openai_key || '',
        groq_key: data.groq_key || '',
        together_key: data.together_key || ''
      }))
      .catch(() => { });
  };

  useEffect(() => {
    checkDownloadedModels();
    fetchStatus();
    fetchCatalog();
    fetchConversations();
    fetchSettings();
  }, []);

  const reconnect = () => {
    fetchStatus();
    fetchCatalog();
    fetchConversations();
    fetchSettings();
  };

  // ── Download model to phone storage ──
  const downloadDeviceModel = async (model: typeof DEVICE_MODELS[0]) => {
    setIsDownloadingDevice(true);
    setDeviceDownloadProgress(0);
    try {
      const dirInfo = await FileSystem.getInfoAsync(modelsDir);
      if (!dirInfo.exists) {
        await FileSystem.makeDirectoryAsync(modelsDir, { intermediates: true });
      }
      const destPath = modelsDir + model.filename;
      const download = FileSystem.createDownloadResumable(
        model.url,
        destPath,
        {},
        (progress) => {
          const pct = progress.totalBytesWritten / progress.totalBytesExpectedToWrite;
          setDeviceDownloadProgress(pct);
        }
      );
      const result = await download.downloadAsync();
      if (result && result.uri) {
        await checkDownloadedModels();
        Alert.alert('Download Complete', `${model.name} is ready to use!`);
      }
    } catch (e: any) {
      Alert.alert('Download Failed', e.message || 'Unknown error');
    } finally {
      setIsDownloadingDevice(false);
    }
  };

  // ── Load model into llama.rn context ──
  const loadDeviceModel = async (model: typeof DEVICE_MODELS[0]) => {
    setIsLoadingDevice(true);
    try {
      // Release previous context if any
      if (llamaContext) {
        await llamaContext.release();
        setLlamaContext(null);
      }
      const modelPath = modelsDir + model.filename;
      const context = await initLlama({
        model: modelPath,
        n_ctx: 2048,
        n_batch: 512,
        n_threads: 4,
        use_mlock: true,
      });
      setLlamaContext(context);
      setDeviceModelName(model.name);
      setChatMode('device');
      setModalVisible(false);
      Alert.alert('Model Loaded', `${model.name} is now running on your phone!`);
    } catch (e: any) {
      Alert.alert('Load Failed', e.message || 'Could not load model. Try a smaller one.');
    } finally {
      setIsLoadingDevice(false);
    }
  };

  // ── Delete a downloaded model ──
  const deleteDeviceModel = async (filename: string) => {
    Alert.alert('Delete Model', `Remove ${filename} from your phone?`, [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Delete', style: 'destructive', onPress: async () => {
          try {
            await FileSystem.deleteAsync(modelsDir + filename);
            await checkDownloadedModels();
          } catch { }
        }
      },
    ]);
  };

  const loadConversation = async (id: string) => {
    try {
      const res = await fetch(`${API_BASE}/api/conversations/${id}`);
      if (res.ok) {
        const data = await res.json();
        setMessages(data.messages || []);
        setCurrentConvId(data.id);
        setSidebarOpen(false);
      }
    } catch (e) { console.error(e); }
  };

  const createNewChat = () => {
    setMessages([{ id: '1', role: 'assistant', content: 'Hello! I am your Personal LLM. Ask me anything!' }]);
    setCurrentConvId(null);
    setSidebarOpen(false);
  };

  const handleSaveSettings = async () => {
    setIsSavingSettings(true);
    try {
      if (isConnected) {
        await fetch(`${API_BASE}/api/settings`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(settings)
        });
      }
      setSettingsOpen(false);
    } catch {
      setSettingsOpen(false);
    } finally {
      setIsSavingSettings(false);
    }
  };

  const handleLoadModel = async (filename: string) => {
    setIsLoadingModel(true);
    try {
      const res = await fetch(`${API_BASE}/api/models/load`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename }),
      });
      if (res.ok) { fetchStatus(); setModalVisible(false); }
      else { Alert.alert('Error', 'Failed to load model.'); }
    } catch { Alert.alert('Error', 'Network error.'); }
    finally { setIsLoadingModel(false); }
  };

  const handleDownloadModel = async (catalogKey: string) => {
    setIsDownloading(true);
    setDownloadProgress(0);
    try {
      const res = await fetch(`${API_BASE}/api/models/download`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ catalog_key: catalogKey }),
      });
      if (!res.ok) { setIsDownloading(false); return; }
      const interval = setInterval(async () => {
        try {
          const s = await fetch(`${API_BASE}/api/models/download/status`);
          const d = await s.json();
          setDownloadProgress(d.progress || 0);
          if (d.done || d.error) { clearInterval(interval); setIsDownloading(false); fetchCatalog(); }
        } catch { clearInterval(interval); setIsDownloading(false); }
      }, 1500);
    } catch { setIsDownloading(false); }
  };

  // ── On-Device Chat (runs entirely on phone) ──
  const handleDeviceChat = async (userMessage: string, asstId: string) => {
    if (!llamaContext) {
      setMessages(prev =>
        prev.map(msg =>
          msg.id === asstId ? { ...msg, content: '⚠️ No model loaded.\n\nTap the 🧠 header → select a 📱 On-Device model → Download → Load.' } : msg
        )
      );
      return;
    }
    try {
      const prompt = `<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n<|im_start|>user\n${userMessage}<|im_end|>\n<|im_start|>assistant\n`;

      const result = await llamaContext.completion({
        prompt,
        n_predict: 512,
        stop: ['<|im_end|>', '<|im_start|>'],
        temperature: 0.7,
      }, (data: any) => {
        // Streaming token callback
        if (data.token) {
          setMessages(prev =>
            prev.map(msg =>
              msg.id === asstId ? { ...msg, content: msg.content + data.token } : msg
            )
          );
        }
      });

      // If no streaming, set final result
      if (result && result.text) {
        setMessages(prev =>
          prev.map(msg =>
            msg.id === asstId && !msg.content ? { ...msg, content: result.text.trim() } : msg
          )
        );
      }
    } catch (e: any) {
      setMessages(prev =>
        prev.map(msg =>
          msg.id === asstId ? { ...msg, content: `⚠️ Inference error: ${e.message}` } : msg
        )
      );
    }
  };

  // ── Cloud Chat (direct from phone) ──
  const handleCloudChat = async (userMessage: string, asstId: string) => {
    const provider = selectedCloudModel.provider;
    const apiKey = provider === 'groq' ? settings.groq_key : settings.openai_key;
    if (!apiKey || apiKey.includes('*')) {
      setMessages(prev =>
        prev.map(msg =>
          msg.id === asstId ? { ...msg, content: `⚠️ No ${provider === 'groq' ? 'Groq' : 'OpenAI'} API key.\n\nGo to ⚙️ Settings and enter your key.\nGroq keys are FREE at console.groq.com` } : msg
        )
      );
      return;
    }
    const endpoint = PROVIDER_ENDPOINTS[provider];
    const chatHistory = messages.filter(m => m.content && !m.content.startsWith('⚠️')).map(m => ({ role: m.role, content: m.content }));
    chatHistory.push({ role: 'user', content: userMessage });
    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${apiKey}` },
        body: JSON.stringify({ model: selectedCloudModel.model, messages: chatHistory.slice(-10), max_tokens: 1024, temperature: 0.7, stream: false }),
      });
      if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        setMessages(prev => prev.map(msg => msg.id === asstId ? { ...msg, content: `⚠️ ${err?.error?.message || `API error ${response.status}`}` } : msg));
        return;
      }
      const data = await response.json();
      setMessages(prev => prev.map(msg => msg.id === asstId ? { ...msg, content: data.choices?.[0]?.message?.content || 'No response' } : msg));
    } catch (e: any) {
      setMessages(prev => prev.map(msg => msg.id === asstId ? { ...msg, content: `⚠️ Network error: ${e.message}` } : msg));
    }
  };

  // ── Local Chat (via Desktop backend) ──
  const handleLocalChat = async (userMessage: string, asstId: string) => {
    try {
      const response = await fetch(`${API_BASE}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userMessage, max_tokens: 1024, temperature: 0.7, conversation_id: currentConvId }),
      });
      if (!response.ok) throw new Error('Stream failed');
      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      while (reader) {
        const { value, done } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value);
        for (const line of chunk.split('\n')) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              if (data.type === 'token') {
                setMessages(prev => prev.map(msg => msg.id === asstId ? { ...msg, content: msg.content + data.content } : msg));
              } else if (data.type === 'done' || data.type === 'error') break;
            } catch { }
          }
        }
      }
    } catch {
      setMessages(prev => prev.map(msg => msg.id === asstId ? { ...msg, content: '⚠️ Cannot connect to Desktop. Check IP in Settings.' } : msg));
    }
  };

  const handleSend = async () => {
    if (!input.trim() || isGenerating) return;
    const userMessage = input;
    const userMsg: Message = { id: Date.now().toString(), role: 'user', content: userMessage };
    const asstId = (Date.now() + 1).toString();
    setMessages(prev => [...prev, userMsg, { id: asstId, role: 'assistant', content: '' }]);
    setInput('');
    setIsGenerating(true);
    if (chatMode === 'device') await handleDeviceChat(userMessage, asstId);
    else if (chatMode === 'cloud') await handleCloudChat(userMessage, asstId);
    else await handleLocalChat(userMessage, asstId);
    setIsGenerating(false);
  };

  const renderMessage = ({ item }: { item: Message }) => {
    const isUser = item.role === 'user';
    return (
      <View style={[styles.msgRow, isUser ? styles.msgRowUser : styles.msgRowBot]}>
        <View style={[styles.avatar, isUser ? styles.avatarUser : styles.avatarBot]}>
          <Text style={styles.avatarText}>{isUser ? '👤' : '🤖'}</Text>
        </View>
        <View style={[styles.bubble, isUser ? styles.bubbleUser : styles.bubbleBot]}>
          {item.content ? (
            <Text style={[styles.msgText, isUser ? styles.msgTextUser : styles.msgTextBot]}>{item.content}</Text>
          ) : isGenerating ? (
            <ActivityIndicator size="small" color="#818cf8" />
          ) : null}
        </View>
      </View>
    );
  };

  const modeLabel = chatMode === 'device'
    ? `📱 ${deviceModelName || 'No Model'}`
    : chatMode === 'cloud'
      ? `☁️ ${selectedCloudModel.name}`
      : (isConnected ? `💻 ${status.name || 'PC'}` : '💻 Not Connected');

  const modeColor = chatMode === 'device' ? '#f59e0b' : chatMode === 'cloud' ? '#22c55e' : '#6366f1';

  return (
    <View style={[styles.safe, { paddingTop: insets.top, paddingBottom: insets.bottom }]}>
      <StatusBar barStyle="light-content" backgroundColor="#0B0E14" translucent={false} />
      <LinearGradient colors={['#0B0E14', '#1a1040', '#0B0E14']} style={styles.container}>

        {/* Header */}
        <View style={styles.header}>
          <View style={styles.headerLeft}>
            <TouchableOpacity onPress={() => setSidebarOpen(true)} style={styles.iconBtn}>
              <Text style={styles.iconText}>☰</Text>
            </TouchableOpacity>
            <TouchableOpacity onPress={() => setModalVisible(true)}>
              <Text style={styles.headerTitle}>🧠 Personal LLM</Text>
              <View style={styles.statusChip}>
                <View style={[styles.statusDot, { backgroundColor: modeColor }]} />
                <Text style={styles.statusText} numberOfLines={1}>{modeLabel}</Text>
              </View>
            </TouchableOpacity>
          </View>
          <TouchableOpacity onPress={() => setSettingsOpen(true)} style={styles.iconBtn}>
            <Text style={styles.iconText}>⚙️</Text>
          </TouchableOpacity>
        </View>

        {/* Chat Mode Toggle */}
        <View style={styles.modeBar}>
          <TouchableOpacity style={[styles.modeBtn, chatMode === 'device' && styles.modeBtnActiveDevice]} onPress={() => setChatMode('device')}>
            <Text style={[styles.modeBtnText, chatMode === 'device' && styles.modeBtnTextActive]}>📱 Device</Text>
          </TouchableOpacity>
          <TouchableOpacity style={[styles.modeBtn, chatMode === 'cloud' && styles.modeBtnActive]} onPress={() => setChatMode('cloud')}>
            <Text style={[styles.modeBtnText, chatMode === 'cloud' && styles.modeBtnTextActive]}>☁️ Cloud</Text>
          </TouchableOpacity>
          <TouchableOpacity style={[styles.modeBtn, chatMode === 'local' && styles.modeBtnActiveLocal]} onPress={() => setChatMode('local')}>
            <Text style={[styles.modeBtnText, chatMode === 'local' && styles.modeBtnTextActive]}>💻 PC</Text>
          </TouchableOpacity>
        </View>

        {/* Sidebar Drawer */}
        <Modal visible={isSidebarOpen} animationType="fade" transparent onRequestClose={() => setSidebarOpen(false)}>
          <View style={styles.sidebarOverlay}>
            <View style={[styles.sidebarContent, { paddingTop: insets.top + 10 }]}>
              <View style={styles.sidebarHeader}>
                <Text style={styles.sidebarTitle}>Chats</Text>
                <TouchableOpacity onPress={() => setSidebarOpen(false)}><Text style={styles.closeModalText}>✕</Text></TouchableOpacity>
              </View>
              <TouchableOpacity style={styles.newChatBtn} onPress={createNewChat}><Text style={styles.newChatText}>+ New Chat</Text></TouchableOpacity>
              <ScrollView style={styles.sidebarScroll}>
                {conversations.map(conv => (
                  <TouchableOpacity key={conv.id} style={[styles.sidebarItem, currentConvId === conv.id && styles.sidebarItemActive]} onPress={() => loadConversation(conv.id)}>
                    <Text style={styles.sidebarItemTitle} numberOfLines={1}>{conv.title}</Text>
                  </TouchableOpacity>
                ))}
                {conversations.length === 0 && <Text style={{ color: '#666', fontSize: 13, textAlign: 'center', marginTop: 30 }}>No chats yet</Text>}
              </ScrollView>
            </View>
            <TouchableOpacity style={styles.sidebarCloseArea} onPress={() => setSidebarOpen(false)} />
          </View>
        </Modal>

        {/* Settings Modal */}
        <Modal visible={isSettingsOpen} animationType="slide" transparent onRequestClose={() => setSettingsOpen(false)}>
          <View style={styles.modalOverlay}>
            <View style={[styles.modalContent, { paddingBottom: insets.bottom + 20 }]}>
              <View style={styles.modalHeader}>
                <Text style={styles.modalTitle}>Settings</Text>
                <TouchableOpacity onPress={() => setSettingsOpen(false)}><Text style={styles.closeModalText}>✕</Text></TouchableOpacity>
              </View>
              <ScrollView style={styles.catalogScroll}>
                <Text style={styles.sectionTitle}>☁️ Cloud API Keys</Text>
                <Text style={styles.settingsDesc}>Get a FREE Groq key at console.groq.com</Text>
                <View style={styles.inputGroup}>
                  <Text style={styles.inputLabel}>Groq API Key (FREE)</Text>
                  <TextInput style={styles.settingsInput} value={settings.groq_key} onChangeText={t => setSettings(p => ({ ...p, groq_key: t }))} placeholder="gsk_..." placeholderTextColor="#555" autoCapitalize="none" />
                </View>
                <View style={styles.inputGroup}>
                  <Text style={styles.inputLabel}>OpenAI API Key</Text>
                  <TextInput style={styles.settingsInput} value={settings.openai_key} onChangeText={t => setSettings(p => ({ ...p, openai_key: t }))} placeholder="sk-..." placeholderTextColor="#555" autoCapitalize="none" />
                </View>
                <Text style={styles.sectionTitle}>💻 Local PC</Text>
                <View style={styles.inputGroup}>
                  <Text style={styles.inputLabel}>Desktop PC IP</Text>
                  <TextInput style={styles.settingsInput} value={backendIp} onChangeText={setBackendIp} placeholder="192.168.0.102" placeholderTextColor="#555" keyboardType="numeric" />
                </View>
                <TouchableOpacity style={styles.reconnectBtn} onPress={reconnect}>
                  <Text style={styles.reconnectBtnText}>🔄 Test Connection</Text>
                </TouchableOpacity>
                <Text style={{ color: isConnected ? '#22c55e' : '#ef4444', fontSize: 13, textAlign: 'center', marginTop: 8 }}>
                  {isConnected ? '✅ Connected' : '❌ Not Connected'}
                </Text>
                <TouchableOpacity style={[styles.saveBtn, isSavingSettings && styles.loadBtnDisabled]} onPress={handleSaveSettings} disabled={isSavingSettings}>
                  <Text style={styles.saveBtnText}>{isSavingSettings ? 'Saving...' : 'Save Settings'}</Text>
                </TouchableOpacity>
              </ScrollView>
            </View>
          </View>
        </Modal>

        {/* Model Selection Modal */}
        <Modal visible={isModalVisible} animationType="slide" transparent onRequestClose={() => setModalVisible(false)}>
          <View style={styles.modalOverlay}>
            <View style={[styles.modalContent, { paddingBottom: insets.bottom + 20 }]}>
              <View style={styles.modalHeader}>
                <Text style={styles.modalTitle}>Select Model</Text>
                <TouchableOpacity onPress={() => setModalVisible(false)}><Text style={styles.closeModalText}>✕</Text></TouchableOpacity>
              </View>
              <ScrollView style={styles.catalogScroll}>

                {/* On-Device Models */}
                <Text style={styles.sectionTitle}>📱 On-Device Models (runs on your phone)</Text>
                <Text style={styles.settingsDesc}>Download once, works completely offline. No internet or PC needed!</Text>
                {DEVICE_MODELS.map(model => {
                  const isOnPhone = downloadedModels.includes(model.filename);
                  const isActive = llamaContext && deviceModelName === model.name;
                  return (
                    <View key={model.id} style={[styles.modelCardDownloaded, isActive && { borderColor: '#f59e0b', borderWidth: 2 }]}>
                      <View style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' }}>
                        <Text style={styles.modelName}>{model.name}</Text>
                        <Text style={styles.sizeBadge}>{model.size_mb < 1000 ? `${model.size_mb} MB` : `${(model.size_mb / 1024).toFixed(1)} GB`}</Text>
                      </View>
                      <Text style={styles.modelDesc}>{model.description}</Text>
                      {isActive && <Text style={{ color: '#f59e0b', fontSize: 12, fontWeight: 'bold', marginTop: 4 }}>✓ Running on device</Text>}
                      {isOnPhone ? (
                        <View style={{ flexDirection: 'row', gap: 8, marginTop: 8 }}>
                          <TouchableOpacity
                            style={[styles.loadBtn, { flex: 1 }, (isLoadingDevice) && styles.loadBtnDisabled]}
                            onPress={() => loadDeviceModel(model)}
                            disabled={isLoadingDevice}
                          >
                            <Text style={styles.loadBtnText}>{isLoadingDevice ? '⏳ Loading...' : '▶ Load & Run'}</Text>
                          </TouchableOpacity>
                          <TouchableOpacity style={styles.deleteBtn} onPress={() => deleteDeviceModel(model.filename)}>
                            <Text style={styles.deleteBtnText}>🗑</Text>
                          </TouchableOpacity>
                        </View>
                      ) : (
                        <TouchableOpacity
                          style={[styles.downloadBtn, isDownloadingDevice && styles.loadBtnDisabled]}
                          onPress={() => downloadDeviceModel(model)}
                          disabled={isDownloadingDevice}
                        >
                          <Text style={styles.downloadBtnText}>
                            {isDownloadingDevice ? `⬇ ${(deviceDownloadProgress * 100).toFixed(0)}%` : `⬇ Download (${model.size_mb < 1000 ? `${model.size_mb} MB` : `${(model.size_mb / 1024).toFixed(1)} GB`})`}
                          </Text>
                        </TouchableOpacity>
                      )}
                      {isDownloadingDevice && (
                        <View style={styles.progressBarBg}>
                          <View style={[styles.progressBarFill, { width: `${Math.max(deviceDownloadProgress * 100, 2)}%` as any }]} />
                        </View>
                      )}
                    </View>
                  );
                })}

                {/* Cloud Models */}
                <Text style={styles.sectionTitle}>☁️ Cloud Models (via API)</Text>
                <Text style={styles.settingsDesc}>Fast inference. Groq models are FREE!</Text>
                {CLOUD_MODELS.map(model => (
                  <TouchableOpacity
                    key={model.id}
                    style={[styles.modelCardAvailable, selectedCloudModel.id === model.id && chatMode === 'cloud' && { borderColor: '#22c55e', borderWidth: 2 }]}
                    onPress={() => { setSelectedCloudModel(model); setChatMode('cloud'); setModalVisible(false); }}
                  >
                    <View style={{ flexDirection: 'row', justifyContent: 'space-between' }}>
                      <Text style={styles.modelName}>{model.name}</Text>
                      {model.free && <View style={styles.freeBadge}><Text style={styles.freeBadgeText}>FREE</Text></View>}
                    </View>
                    <Text style={styles.modelDesc}>{model.provider === 'groq' ? 'Ultra-fast via Groq' : 'OpenAI cloud'}</Text>
                  </TouchableOpacity>
                ))}

                {/* PC Models */}
                {isConnected && (
                  <>
                    <Text style={styles.sectionTitle}>💻 Local PC Models</Text>
                    {catalog.filter(c => c.is_downloaded).map(model => (
                      <TouchableOpacity key={model.key} style={styles.modelCardAvailable} onPress={() => { handleLoadModel(model.filename); setChatMode('local'); }}>
                        <Text style={styles.modelName}>{model.name}</Text>
                        <Text style={styles.modelDesc}>{model.size_gb} GB • Local</Text>
                      </TouchableOpacity>
                    ))}
                    {catalog.filter(c => !c.is_downloaded).map(model => (
                      <View key={model.key} style={styles.modelCardAvailable}>
                        <Text style={styles.modelName}>{model.name}</Text>
                        <Text style={styles.modelMeta}>{model.size_gb} GB</Text>
                        <TouchableOpacity style={[styles.downloadBtn, isDownloading && styles.loadBtnDisabled]} onPress={() => handleDownloadModel(model.key)} disabled={isDownloading}>
                          <Text style={styles.downloadBtnText}>{isDownloading ? `${(downloadProgress * 100).toFixed(0)}%` : `⬇ Download to PC`}</Text>
                        </TouchableOpacity>
                      </View>
                    ))}
                  </>
                )}
              </ScrollView>
            </View>
          </View>
        </Modal>

        {/* Messages */}
        <FlatList
          ref={flatListRef}
          data={messages}
          renderItem={renderMessage}
          keyExtractor={item => item.id}
          contentContainerStyle={styles.chatList}
          onContentSizeChange={() => flatListRef.current?.scrollToEnd({ animated: true })}
        />

        {/* Input */}
        <KeyboardAvoidingView behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
          <View style={styles.inputBar}>
            <TextInput
              style={styles.textInput}
              value={input}
              onChangeText={setInput}
              placeholder="Type a message..."
              placeholderTextColor="#555"
              multiline
              maxLength={2000}
              editable={!isGenerating}
            />
            <TouchableOpacity
              style={[styles.sendBtn, (!input.trim() || isGenerating) && styles.sendBtnDisabled]}
              onPress={handleSend}
              disabled={!input.trim() || isGenerating}
            >
              <Text style={styles.sendBtnText}>▶</Text>
            </TouchableOpacity>
          </View>
        </KeyboardAvoidingView>
      </LinearGradient>
    </View>
  );
}

export default function App() {
  return (
    <SafeAreaProvider>
      <AppContent />
    </SafeAreaProvider>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: '#0B0E14' },
  container: { flex: 1 },
  header: { paddingHorizontal: 16, paddingTop: 8, paddingBottom: 8, borderBottomWidth: 1, borderBottomColor: 'rgba(255,255,255,0.06)', flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  headerLeft: { flexDirection: 'row', alignItems: 'center', flex: 1 },
  iconBtn: { padding: 8 },
  iconText: { color: '#fff', fontSize: 22 },
  headerTitle: { color: '#fff', fontSize: 17, fontWeight: '800', marginLeft: 8 },
  statusChip: { flexDirection: 'row', alignItems: 'center', marginTop: 2, marginLeft: 8 },
  statusDot: { width: 6, height: 6, borderRadius: 3, marginRight: 6 },
  statusText: { color: '#aaa', fontSize: 11, maxWidth: 200 },
  modeBar: { flexDirection: 'row', paddingHorizontal: 12, paddingVertical: 6, borderBottomWidth: 1, borderBottomColor: 'rgba(255,255,255,0.06)', gap: 6 },
  modeBtn: { flex: 1, paddingVertical: 7, borderRadius: 10, backgroundColor: 'rgba(255,255,255,0.05)', alignItems: 'center', borderWidth: 1, borderColor: 'rgba(255,255,255,0.08)' },
  modeBtnActive: { backgroundColor: 'rgba(34,197,94,0.15)', borderColor: '#22c55e' },
  modeBtnActiveDevice: { backgroundColor: 'rgba(245,158,11,0.15)', borderColor: '#f59e0b' },
  modeBtnActiveLocal: { backgroundColor: 'rgba(99,102,241,0.15)', borderColor: '#6366f1' },
  modeBtnText: { color: '#888', fontSize: 12, fontWeight: '600' },
  modeBtnTextActive: { color: '#fff' },
  chatList: { paddingHorizontal: 16, paddingVertical: 12 },
  msgRow: { flexDirection: 'row', marginBottom: 16, alignItems: 'flex-end' },
  msgRowUser: { flexDirection: 'row-reverse' },
  msgRowBot: { flexDirection: 'row' },
  avatar: { width: 32, height: 32, borderRadius: 10, alignItems: 'center', justifyContent: 'center', marginHorizontal: 6 },
  avatarUser: { backgroundColor: '#4f46e5' },
  avatarBot: { backgroundColor: '#1e2330', borderWidth: 1, borderColor: 'rgba(255,255,255,0.1)' },
  avatarText: { fontSize: 14 },
  bubble: { maxWidth: '75%', padding: 12, borderRadius: 16 },
  bubbleUser: { backgroundColor: '#4f46e5', borderBottomRightRadius: 4 },
  bubbleBot: { backgroundColor: 'rgba(30,35,48,0.8)', borderBottomLeftRadius: 4, borderWidth: 1, borderColor: 'rgba(255,255,255,0.06)' },
  msgText: { fontSize: 14, lineHeight: 20 },
  msgTextUser: { color: '#fff' },
  msgTextBot: { color: '#e2e8f0' },
  inputBar: { flexDirection: 'row', alignItems: 'flex-end', paddingHorizontal: 12, paddingTop: 8, paddingBottom: 8, borderTopWidth: 1, borderTopColor: 'rgba(255,255,255,0.06)', backgroundColor: 'rgba(11,14,20,0.95)' },
  textInput: { flex: 1, backgroundColor: '#151923', color: '#fff', borderRadius: 16, paddingHorizontal: 16, paddingVertical: 10, fontSize: 14, maxHeight: 100, borderWidth: 1, borderColor: 'rgba(255,255,255,0.08)' },
  sendBtn: { width: 42, height: 42, borderRadius: 14, backgroundColor: '#4f46e5', alignItems: 'center', justifyContent: 'center', marginLeft: 8 },
  sendBtnDisabled: { backgroundColor: '#333' },
  sendBtnText: { color: '#fff', fontSize: 16 },
  modalOverlay: { flex: 1, backgroundColor: 'rgba(0,0,0,0.7)', justifyContent: 'flex-end' },
  modalContent: { backgroundColor: '#0B0E14', borderTopLeftRadius: 24, borderTopRightRadius: 24, height: '88%', padding: 20, borderWidth: 1, borderColor: 'rgba(255,255,255,0.1)' },
  modalHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16, borderBottomWidth: 1, borderBottomColor: 'rgba(255,255,255,0.1)', paddingBottom: 12 },
  modalTitle: { color: '#fff', fontSize: 18, fontWeight: 'bold' },
  closeModalText: { color: '#aaa', fontSize: 20, padding: 5 },
  catalogScroll: { flex: 1 },
  sectionTitle: { color: '#8b5cf6', fontSize: 13, fontWeight: 'bold', textTransform: 'uppercase', marginTop: 20, marginBottom: 10, letterSpacing: 1 },
  modelCardDownloaded: { backgroundColor: 'rgba(79,70,229,0.1)', borderWidth: 1, borderColor: 'rgba(79,70,229,0.3)', borderRadius: 14, padding: 14, marginBottom: 10 },
  modelCardAvailable: { backgroundColor: 'rgba(255,255,255,0.05)', borderRadius: 14, padding: 14, marginBottom: 10, borderWidth: 1, borderColor: 'transparent' },
  modelName: { color: '#fff', fontSize: 15, fontWeight: 'bold', marginBottom: 2 },
  modelDesc: { color: '#aaa', fontSize: 12, lineHeight: 16, marginBottom: 4 },
  modelMeta: { color: '#6366f1', fontSize: 11, fontWeight: 'bold', marginBottom: 6 },
  sizeBadge: { color: '#f59e0b', fontSize: 11, fontWeight: 'bold', backgroundColor: 'rgba(245,158,11,0.15)', paddingHorizontal: 8, paddingVertical: 2, borderRadius: 6 },
  freeBadge: { backgroundColor: '#22c55e', paddingHorizontal: 8, paddingVertical: 2, borderRadius: 6 },
  freeBadgeText: { color: '#fff', fontSize: 10, fontWeight: 'bold' },
  loadBtn: { backgroundColor: '#4f46e5', paddingVertical: 8, borderRadius: 10, alignItems: 'center' },
  loadBtnDisabled: { opacity: 0.5 },
  loadBtnText: { color: '#fff', fontWeight: 'bold', fontSize: 13 },
  downloadBtn: { backgroundColor: '#22c55e', paddingVertical: 8, borderRadius: 10, alignItems: 'center', marginTop: 6 },
  downloadBtnText: { color: '#fff', fontWeight: 'bold', fontSize: 13 },
  deleteBtn: { backgroundColor: 'rgba(239,68,68,0.2)', paddingVertical: 8, paddingHorizontal: 14, borderRadius: 10, alignItems: 'center', justifyContent: 'center' },
  deleteBtnText: { fontSize: 16 },
  progressBarBg: { height: 5, backgroundColor: 'rgba(255,255,255,0.1)', borderRadius: 3, marginTop: 6, overflow: 'hidden' as const },
  progressBarFill: { height: 5, backgroundColor: '#22c55e', borderRadius: 3 },
  reconnectBtn: { backgroundColor: '#4f46e5', paddingVertical: 10, paddingHorizontal: 24, borderRadius: 10, alignItems: 'center', marginTop: 12 },
  reconnectBtnText: { color: '#fff', fontWeight: 'bold', fontSize: 14 },
  sidebarOverlay: { flex: 1, flexDirection: 'row', backgroundColor: 'rgba(0,0,0,0.5)' },
  sidebarContent: { width: '80%', maxWidth: 320, backgroundColor: '#0B0E14', height: '100%', padding: 20, borderRightWidth: 1, borderColor: 'rgba(255,255,255,0.1)' },
  sidebarCloseArea: { flex: 1 },
  sidebarHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 },
  sidebarTitle: { color: '#fff', fontSize: 18, fontWeight: 'bold' },
  newChatBtn: { backgroundColor: '#4f46e5', padding: 12, borderRadius: 12, alignItems: 'center', marginBottom: 16 },
  newChatText: { color: '#fff', fontWeight: 'bold' },
  sidebarScroll: { flex: 1 },
  sidebarItem: { paddingVertical: 12, borderBottomWidth: 1, borderBottomColor: 'rgba(255,255,255,0.05)' },
  sidebarItemActive: { backgroundColor: 'rgba(79,70,229,0.1)', paddingHorizontal: 12, borderRadius: 8, borderBottomWidth: 0 },
  sidebarItemTitle: { color: '#fff', fontSize: 14, fontWeight: '500' },
  settingsDesc: { color: '#aaa', fontSize: 12, marginBottom: 14, lineHeight: 18 },
  inputGroup: { marginBottom: 14 },
  inputLabel: { color: '#ccc', fontSize: 13, marginBottom: 6, fontWeight: '500' },
  settingsInput: { backgroundColor: '#151923', color: '#fff', borderRadius: 12, paddingHorizontal: 14, paddingVertical: 10, fontSize: 14, borderWidth: 1, borderColor: 'rgba(255,255,255,0.08)' },
  saveBtn: { backgroundColor: '#22c55e', paddingVertical: 12, borderRadius: 12, alignItems: 'center', marginTop: 20, marginBottom: 40 },
  saveBtnText: { color: '#fff', fontWeight: 'bold', fontSize: 15 },
});
