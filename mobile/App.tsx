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
  Image,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { SafeAreaProvider, useSafeAreaInsets } from 'react-native-safe-area-context';
import { initLlama, type LlamaContext } from 'llama.rn';
import * as FileSystem from 'expo-file-system/legacy';
import * as ImagePicker from 'expo-image-picker';

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

// ─── Cloud Models (Gemini & Claude) ───
const CLOUD_MODELS = [
  { id: 'gemini-flash', name: 'Gemini 2.0 Flash', provider: 'gemini', model: 'gemini-2.0-flash', description: 'Google\'s fastest model. Great for everyday tasks and vision.' },
  { id: 'gemini-pro', name: 'Gemini 1.5 Pro', provider: 'gemini', model: 'gemini-1.5-pro', description: 'Google\'s most capable model for complex reasoning and vision.' },
  { id: 'claude-sonnet', name: 'Claude 3.5 Sonnet', provider: 'claude', model: 'claude-3-5-sonnet-20241022', description: 'Anthropic\'s best all-around model (vision supported).' },
  { id: 'claude-haiku', name: 'Claude 3.5 Haiku', provider: 'claude', model: 'claude-3-5-haiku-20241022', description: 'Fast and affordable Claude model.' },
];

// ─── Provider API endpoints ───
const PROVIDER_CONFIG: Record<string, { url: string; format: 'openai' | 'anthropic' }> = {
  gemini: {
    url: 'https://generativelanguage.googleapis.com/v1beta/openai/chat/completions',
    format: 'openai',
  },
  claude: {
    url: 'https://api.anthropic.com/v1/messages',
    format: 'anthropic',
  },
};

type Role = 'user' | 'assistant';
type Message = { id: string; role: Role; content: string; imageBase64?: string; imageUri?: string; mimeType?: string };
type ChatMode = 'device' | 'cloud';

export function AppContent() {
  const insets = useSafeAreaInsets();

  const [messages, setMessages] = useState<Message[]>([
    { id: '1', role: 'assistant', content: 'Hello! I am your Personal LLM. Ask me anything, or send me a photo!' },
  ]);
  const [input, setInput] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [isModalVisible, setModalVisible] = useState(false);

  // Attachment state
  const [attachedImage, setAttachedImage] = useState<{ uri: string; base64: string; mimeType: string } | null>(null);

  // Chat mode
  const [chatMode, setChatMode] = useState<ChatMode>('device');
  const [selectedCloudModel, setSelectedCloudModel] = useState(CLOUD_MODELS[0]);

  // On-device state
  const [llamaContext, setLlamaContext] = useState<LlamaContext | null>(null);
  const [deviceModelName, setDeviceModelName] = useState('');
  const [downloadingModelId, setDownloadingModelId] = useState<string | null>(null);
  const [deviceDownloadProgress, setDeviceDownloadProgress] = useState(0);
  const [loadingModelId, setLoadingModelId] = useState<string | null>(null);
  const [downloadedModels, setDownloadedModels] = useState<string[]>([]);

  // UI States
  const [isSidebarOpen, setSidebarOpen] = useState(false);
  const [isSettingsOpen, setSettingsOpen] = useState(false);

  // Settings
  const [settings, setSettings] = useState({ gemini_key: '', claude_key: '' });

  const flatListRef = useRef<FlatList>(null);
  const modelsDir = `${FileSystem.cacheDirectory}models/`;

  const checkDownloadedModels = async () => {
    try {
      const dirInfo = await FileSystem.getInfoAsync(modelsDir);
      if (!dirInfo.exists) {
        await FileSystem.makeDirectoryAsync(modelsDir, { intermediates: true });
        setDownloadedModels([]);
        return;
      }
      const files = await FileSystem.readDirectoryAsync(modelsDir);
      setDownloadedModels(files.filter((f: string) => f.endsWith('.gguf')));
    } catch {
      setDownloadedModels([]);
    }
  };

  useEffect(() => {
    checkDownloadedModels();
  }, []);

  const downloadDeviceModel = async (model: typeof DEVICE_MODELS[0]) => {
    setDownloadingModelId(model.id);
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
        Alert.alert('Download Complete', `${model.name} is ready! Tap "Load & Run".`);
      }
    } catch (e: any) {
      Alert.alert('Download Failed', e.message || 'Unknown error');
    } finally {
      setDownloadingModelId(null);
    }
  };

  const loadDeviceModel = async (model: typeof DEVICE_MODELS[0]) => {
    setLoadingModelId(model.id);
    try {
      if (llamaContext) {
        try {
          await llamaContext.release();
        } catch (e) {
          console.warn('Error releasing previous context:', e);
        } finally {
          setLlamaContext(null); // Always clear the state, even if release throws
        }
        // Wait for native memory to be freed before loading the new model
        await new Promise(r => setTimeout(r, 500));
      }
      
      let modelPath = modelsDir + model.filename;
      if (modelPath.startsWith('file://')) {
        modelPath = modelPath.replace('file://', '');
      }
      
      console.log('[LLM] Loading model from:', modelPath);
      const context = await initLlama({
        model: modelPath,
        n_ctx: 2048,
        n_batch: 512,
        n_threads: 4,
        use_mlock: false, // safer for android
      });
      setLlamaContext(context);
      setDeviceModelName(model.name);
      setChatMode('device');
      setModalVisible(false);
      Alert.alert('Model Loaded', `${model.name} is running!`);
    } catch (e: any) {
      console.error('[LLM] Load error:', e);
      Alert.alert('Load Failed', `${e.message || 'Unknown error'}\n\nPath: ${modelsDir + model.filename}\n\nTry deleting and re-downloading.`);
    } finally {
      setLoadingModelId(null);
    }
  };

  const deleteDeviceModel = async (filename: string) => {
    Alert.alert('Delete Model', `Remove from phone?`, [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Delete', style: 'destructive', onPress: async () => {
          try {
            await FileSystem.deleteAsync(modelsDir + filename);
            await checkDownloadedModels();
            if (llamaContext && deviceModelName === DEVICE_MODELS.find(m => m.filename === filename)?.name) {
              try { await llamaContext.release(); } catch {}
              setLlamaContext(null);
              setDeviceModelName('');
            }
          } catch {}
        }
      },
    ]);
  };

  const createNewChat = () => {
    setMessages([{ id: '1', role: 'assistant', content: 'Hello! I am your Personal LLM. Ask me anything!' }]);
    setSidebarOpen(false);
  };

  // ── Image Picker ──
  const pickImage = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'],
      allowsEditing: true,
      quality: 0.5,
      base64: true,
    });

    if (!result.canceled && result.assets[0].base64) {
      const asset = result.assets[0];
      setAttachedImage({
        uri: asset.uri,
        base64: asset.base64 as string,
        mimeType: asset.mimeType || 'image/jpeg',
      });
    }
  };

  const clearAttachment = () => setAttachedImage(null);

  // ── On-Device Chat ──
  const handleDeviceChat = async (userMessage: string, asstId: string, hasImage: boolean) => {
    if (!llamaContext) {
      setMessages(prev => prev.map(msg => msg.id === asstId ? { ...msg, content: '⚠️ No model loaded. Tap the 🧠 header.' } : msg));
      return;
    }
    if (hasImage) {
      // On-device models don't support vision — append a note but still process the text
      const imageNote = '⚠️ Note: The on-device model cannot analyze images. Responding to your text only.\n\n';
      try {
        const prompt = `<|im_start|>system\nYou are a helpful AI assistant. The user tried to share an image, but this model does not support vision. Respond helpfully to their text message.<|im_end|>\n<|im_start|>user\n${userMessage}<|im_end|>\n<|im_start|>assistant\n`;

        setMessages(prev => prev.map(msg => msg.id === asstId ? { ...msg, content: imageNote } : msg));

        const result = await llamaContext.completion({
          prompt,
          n_predict: 512,
          stop: ['<|im_end|>', '<|im_start|>'],
          temperature: 0.7,
        }, (data: any) => {
          if (data.token) {
            setMessages(prev => prev.map(msg => msg.id === asstId ? { ...msg, content: imageNote + msg.content.slice(imageNote.length) + data.token } : msg));
          }
        });

        if (result && result.text) {
          setMessages(prev => prev.map(msg => msg.id === asstId && msg.content === imageNote ? { ...msg, content: imageNote + result.text.trim() } : msg));
        }
      } catch (e: any) {
        setMessages(prev => prev.map(msg => msg.id === asstId ? { ...msg, content: `⚠️ Inference error: ${e.message}` } : msg));
      }
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
        if (data.token) {
          setMessages(prev => prev.map(msg => msg.id === asstId ? { ...msg, content: msg.content + data.token } : msg));
        }
      });

      if (result && result.text) {
        setMessages(prev => prev.map(msg => msg.id === asstId && !msg.content ? { ...msg, content: result.text.trim() } : msg));
      }
    } catch (e: any) {
      setMessages(prev => prev.map(msg => msg.id === asstId ? { ...msg, content: `⚠️ Inference error: ${e.message}` } : msg));
    }
  };

  // ── Cloud Chat ──
  const handleCloudChat = async (userMessage: string, asstId: string, imageObj: typeof attachedImage) => {
    const provider = selectedCloudModel.provider;
    const apiKey = provider === 'gemini' ? settings.gemini_key : settings.claude_key;
    if (!apiKey || apiKey.includes('*')) {
      setMessages(prev => prev.map(msg => msg.id === asstId ? { ...msg, content: `⚠️ No ${provider} API key configured.` } : msg));
      return;
    }

    const config = PROVIDER_CONFIG[provider];
    
    // Format chat history for provider constraints
    const formattedMessages = messages
      .filter(m => m.content && !m.content.startsWith('⚠️'))
      .map(m => {
        // Transform past messages with images if needed
        if (m.imageBase64 && m.mimeType) {
          if (config.format === 'openai') {
            return {
              role: m.role,
              content: [
                { type: 'text', text: m.content || " " },
                { type: 'image_url', image_url: { url: `data:${m.mimeType};base64,${m.imageBase64}` } }
              ]
            };
          } else if (config.format === 'anthropic') {
            return {
              role: m.role,
              content: [
                { type: 'image', source: { type: 'base64', media_type: m.mimeType, data: m.imageBase64 } },
                { type: 'text', text: m.content || " " }
              ]
            };
          }
        }
        return { role: m.role, content: m.content };
      });

    // Handle new message
    let newUserMessageContent: any = userMessage || " ";
    if (imageObj) {
      if (config.format === 'openai') {
        newUserMessageContent = [
          { type: 'text', text: userMessage || " " },
          { type: 'image_url', image_url: { url: `data:${imageObj.mimeType};base64,${imageObj.base64}` } }
        ];
      } else if (config.format === 'anthropic') {
        newUserMessageContent = [
          { type: 'image', source: { type: 'base64', media_type: imageObj.mimeType, data: imageObj.base64 } },
          { type: 'text', text: userMessage || " " }
        ];
      }
    }
    
    formattedMessages.push({ role: 'user', content: newUserMessageContent });

    try {
      let bodyData: any = {};
      let headersConfig: any = { 'Content-Type': 'application/json' };

      if (config.format === 'openai') {
        headersConfig['Authorization'] = `Bearer ${apiKey}`;
        bodyData = {
          model: selectedCloudModel.model,
          messages: formattedMessages.slice(-10),
          max_tokens: 1024,
          temperature: 0.7,
        };
      } else if (config.format === 'anthropic') {
        headersConfig['x-api-key'] = apiKey;
        headersConfig['anthropic-version'] = '2023-06-01';
        bodyData = {
          model: selectedCloudModel.model,
          messages: formattedMessages.slice(-10),
          max_tokens: 1024,
        };
      }

      const response = await fetch(config.url, {
        method: 'POST',
        headers: headersConfig,
        body: JSON.stringify(bodyData),
      });

      if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        setMessages(prev => prev.map(msg => msg.id === asstId ? { ...msg, content: `⚠️ API error: ${err?.error?.message || response.status}` } : msg));
        return;
      }
      
      const data = await response.json();
      let responseText = 'No response';
      if (config.format === 'openai') responseText = data.choices?.[0]?.message?.content || 'No response';
      else if (config.format === 'anthropic') responseText = data.content?.[0]?.text || 'No response';

      setMessages(prev => prev.map(msg => msg.id === asstId ? { ...msg, content: responseText } : msg));
    } catch (e: any) {
      setMessages(prev => prev.map(msg => msg.id === asstId ? { ...msg, content: `⚠️ Network error: ${e.message}` } : msg));
    }
  };

  const handleSend = async () => {
    if ((!input.trim() && !attachedImage) || isGenerating) return;
    
    const userMessage = input;
    const currentAttachment = attachedImage;
    
    const userMsg: Message = { 
      id: Date.now().toString(), 
      role: 'user', 
      content: userMessage,
      imageBase64: currentAttachment?.base64,
      imageUri: currentAttachment?.uri,
      mimeType: currentAttachment?.mimeType
    };
    
    const asstId = (Date.now() + 1).toString();
    setMessages(prev => [...prev, userMsg, { id: asstId, role: 'assistant', content: '' }]);
    setInput('');
    setAttachedImage(null);
    setIsGenerating(true);
    
    if (chatMode === 'device') {
      await handleDeviceChat(userMessage, asstId, !!currentAttachment);
    } else {
      await handleCloudChat(userMessage, asstId, currentAttachment);
    }
    
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
          {item.imageUri && (
            <Image source={{ uri: item.imageUri }} style={styles.msgImage} />
          )}
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
    : `☁️ ${selectedCloudModel.name}`;
  const modeColor = chatMode === 'device' ? '#f59e0b' : '#22c55e';

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
        </View>

        {/* Sidebar */}
        <Modal visible={isSidebarOpen} animationType="fade" transparent onRequestClose={() => setSidebarOpen(false)}>
          <View style={styles.sidebarOverlay}>
            <View style={[styles.sidebarContent, { paddingTop: insets.top + 10 }]}>
              <View style={styles.sidebarHeader}>
                <Text style={styles.sidebarTitle}>Chats</Text>
                <TouchableOpacity onPress={() => setSidebarOpen(false)}><Text style={styles.closeModalText}>✕</Text></TouchableOpacity>
              </View>
              <TouchableOpacity style={styles.newChatBtn} onPress={createNewChat}><Text style={styles.newChatText}>+ New Chat</Text></TouchableOpacity>
            </View>
            <TouchableOpacity style={styles.sidebarCloseArea} onPress={() => setSidebarOpen(false)} />
          </View>
        </Modal>

        {/* Settings Modal */}
        <Modal visible={isSettingsOpen} animationType="slide" transparent>
          <View style={styles.modalOverlay}>
            <View style={[styles.modalContent, { paddingBottom: insets.bottom + 20 }]}>
              <View style={styles.modalHeader}>
                <Text style={styles.modalTitle}>Settings</Text>
                <TouchableOpacity onPress={() => setSettingsOpen(false)}><Text style={styles.closeModalText}>✕</Text></TouchableOpacity>
              </View>
              <ScrollView>
                <Text style={styles.sectionTitle}>☁️ Cloud API Keys</Text>
                <View style={styles.inputGroup}>
                  <Text style={styles.inputLabel}>🔷 Gemini API Key</Text>
                  <TextInput style={styles.settingsInput} value={settings.gemini_key} onChangeText={t => setSettings(p => ({ ...p, gemini_key: t }))} placeholder="AIza..." placeholderTextColor="#555" autoCapitalize="none" />
                </View>
                <View style={styles.inputGroup}>
                  <Text style={styles.inputLabel}>🟠 Claude API Key</Text>
                  <TextInput style={styles.settingsInput} value={settings.claude_key} onChangeText={t => setSettings(p => ({ ...p, claude_key: t }))} placeholder="sk-ant-..." placeholderTextColor="#555" autoCapitalize="none" />
                </View>
                <TouchableOpacity style={styles.saveBtn} onPress={() => setSettingsOpen(false)}><Text style={styles.saveBtnText}>Save</Text></TouchableOpacity>
              </ScrollView>
            </View>
          </View>
        </Modal>

        {/* Model Selection Modal */}
        <Modal visible={isModalVisible} animationType="slide" transparent>
          <View style={styles.modalOverlay}>
            <View style={[styles.modalContent, { paddingBottom: insets.bottom + 20 }]}>
              <View style={styles.modalHeader}>
                <Text style={styles.modalTitle}>Select Model</Text>
                <TouchableOpacity onPress={() => setModalVisible(false)}><Text style={styles.closeModalText}>✕</Text></TouchableOpacity>
              </View>
              <ScrollView style={styles.catalogScroll}>
                <Text style={styles.sectionTitle}>📱 On-Device Models</Text>
                {DEVICE_MODELS.map(model => {
                  const isOnPhone = downloadedModels.includes(model.filename);
                  const isActive = llamaContext && deviceModelName === model.name;
                  const isThisLoading = loadingModelId === model.id;
                  return (
                    <View key={model.id} style={[styles.modelCardDownloaded, isActive && { borderColor: '#f59e0b', borderWidth: 2 }]}>
                      <View style={{ flexDirection: 'row', justifyContent: 'space-between' }}>
                        <Text style={styles.modelName}>{model.name}</Text>
                        <Text style={styles.sizeBadge}>{model.size_mb < 1000 ? `${model.size_mb} MB` : `${(model.size_mb / 1024).toFixed(1)} GB`}</Text>
                      </View>
                      <Text style={styles.modelDesc}>{model.description}</Text>
                      {isActive && <Text style={{ color: '#f59e0b', fontSize: 12, fontWeight: 'bold' }}>✓ Running</Text>}
                      {isOnPhone ? (
                        <View style={{ flexDirection: 'row', gap: 8, marginTop: 8 }}>
                          <TouchableOpacity style={[styles.loadBtn, { flex: 1 }, isThisLoading && styles.loadBtnDisabled]} onPress={() => loadDeviceModel(model)} disabled={isThisLoading}>
                            <Text style={styles.loadBtnText}>{isThisLoading ? '⏳ Loading...' : '▶ Load & Run'}</Text>
                          </TouchableOpacity>
                          <TouchableOpacity style={styles.deleteBtn} onPress={() => deleteDeviceModel(model.filename)}><Text style={styles.deleteBtnText}>🗑</Text></TouchableOpacity>
                        </View>
                      ) : (
                        <TouchableOpacity style={[styles.downloadBtn, downloadingModelId === model.id && styles.loadBtnDisabled]} onPress={() => downloadDeviceModel(model)} disabled={downloadingModelId !== null}>
                          <Text style={styles.downloadBtnText}>{downloadingModelId === model.id ? `⬇ ${(deviceDownloadProgress * 100).toFixed(0)}%` : `⬇ Download`}</Text>
                        </TouchableOpacity>
                      )}
                    </View>
                  );
                })}

                <Text style={styles.sectionTitle}>☁️ Cloud Models</Text>
                {CLOUD_MODELS.map(model => {
                  const isSelected = selectedCloudModel.id === model.id && chatMode === 'cloud';
                  return (
                    <TouchableOpacity key={model.id} style={[styles.modelCardAvailable, isSelected && { borderColor: '#22c55e', borderWidth: 2 }]} onPress={() => { setSelectedCloudModel(model); setChatMode('cloud'); setModalVisible(false); }}>
                      <Text style={styles.modelName}>{model.name}</Text>
                      <Text style={styles.modelDesc}>{model.description}</Text>
                    </TouchableOpacity>
                  );
                })}
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

        {/* Input Area with Attachment Preview */}
        <KeyboardAvoidingView behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
          {attachedImage && (
            <View style={styles.attachmentPreviewContainer}>
              <View style={styles.attachmentBadge}>
                <Image source={{ uri: attachedImage.uri }} style={styles.attachmentImage} />
                <TouchableOpacity style={styles.attachmentClear} onPress={clearAttachment}>
                  <Text style={{color: 'white', fontSize: 12, fontWeight: 'bold'}}>X</Text>
                </TouchableOpacity>
              </View>
            </View>
          )}
          <View style={styles.inputBar}>
            <TouchableOpacity onPress={pickImage} style={styles.attachBtn}>
              <Text style={styles.attachBtnText}>📎</Text>
            </TouchableOpacity>
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
              style={[styles.sendBtn, (!input.trim() && !attachedImage || isGenerating) && styles.sendBtnDisabled]}
              onPress={handleSend}
              disabled={(!input.trim() && !attachedImage) || isGenerating}
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
  msgImage: { width: 200, height: 200, borderRadius: 12, marginBottom: 8, resizeMode: 'cover' },
  inputBar: { flexDirection: 'row', alignItems: 'flex-end', paddingHorizontal: 12, paddingTop: 8, paddingBottom: 8, borderTopWidth: 1, borderTopColor: 'rgba(255,255,255,0.06)', backgroundColor: 'rgba(11,14,20,0.95)' },
  textInput: { flex: 1, backgroundColor: '#151923', color: '#fff', borderRadius: 16, paddingHorizontal: 16, paddingVertical: 10, fontSize: 14, maxHeight: 100, borderWidth: 1, borderColor: 'rgba(255,255,255,0.08)' },
  sendBtn: { width: 42, height: 42, borderRadius: 14, backgroundColor: '#4f46e5', alignItems: 'center', justifyContent: 'center', marginLeft: 8 },
  sendBtnDisabled: { backgroundColor: '#333' },
  sendBtnText: { color: '#fff', fontSize: 16 },
  attachBtn: { width: 42, height: 42, borderRadius: 14, backgroundColor: 'rgba(255,255,255,0.05)', alignItems: 'center', justifyContent: 'center', marginRight: 8 },
  attachBtnText: { color: '#aaa', fontSize: 18 },
  attachmentPreviewContainer: { paddingHorizontal: 16, paddingVertical: 8, backgroundColor: 'rgba(11,14,20,0.95)', borderTopWidth: 1, borderTopColor: 'rgba(255,255,255,0.06)' },
  attachmentBadge: { width: 60, height: 60, borderRadius: 8, backgroundColor: '#151923' },
  attachmentImage: { width: '100%', height: '100%', borderRadius: 8, resizeMode: 'cover' },
  attachmentClear: { position: 'absolute', top: -5, right: -5, backgroundColor: '#ef4444', width: 20, height: 20, borderRadius: 10, alignItems: 'center', justifyContent: 'center' },
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
  sizeBadge: { color: '#f59e0b', fontSize: 11, fontWeight: 'bold', backgroundColor: 'rgba(245,158,11,0.15)', paddingHorizontal: 8, paddingVertical: 2, borderRadius: 6 },
  loadBtn: { backgroundColor: '#4f46e5', paddingVertical: 8, borderRadius: 10, alignItems: 'center' },
  loadBtnDisabled: { opacity: 0.5 },
  loadBtnText: { color: '#fff', fontWeight: 'bold', fontSize: 13 },
  downloadBtn: { backgroundColor: '#22c55e', paddingVertical: 8, borderRadius: 10, alignItems: 'center', marginTop: 6 },
  downloadBtnText: { color: '#fff', fontWeight: 'bold', fontSize: 13 },
  deleteBtn: { backgroundColor: 'rgba(239,68,68,0.2)', paddingVertical: 8, paddingHorizontal: 14, borderRadius: 10, alignItems: 'center', justifyContent: 'center' },
  deleteBtnText: { fontSize: 16 },
  sidebarOverlay: { flex: 1, flexDirection: 'row', backgroundColor: 'rgba(0,0,0,0.5)' },
  sidebarContent: { width: '80%', maxWidth: 320, backgroundColor: '#0B0E14', height: '100%', padding: 20, borderRightWidth: 1, borderColor: 'rgba(255,255,255,0.1)' },
  sidebarCloseArea: { flex: 1 },
  sidebarHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 },
  sidebarTitle: { color: '#fff', fontSize: 18, fontWeight: 'bold' },
  newChatBtn: { backgroundColor: '#4f46e5', padding: 12, borderRadius: 12, alignItems: 'center', marginBottom: 16 },
  newChatText: { color: '#fff', fontWeight: 'bold' },
  inputGroup: { marginBottom: 14 },
  inputLabel: { color: '#ccc', fontSize: 13, marginBottom: 4, fontWeight: '500' },
  settingsInput: { backgroundColor: '#151923', color: '#fff', borderRadius: 12, paddingHorizontal: 14, paddingVertical: 10, fontSize: 14, borderWidth: 1, borderColor: 'rgba(255,255,255,0.08)' },
  saveBtn: { backgroundColor: '#22c55e', paddingVertical: 12, borderRadius: 12, alignItems: 'center', marginTop: 20, marginBottom: 40 },
  saveBtnText: { color: '#fff', fontWeight: 'bold', fontSize: 15 },
});
