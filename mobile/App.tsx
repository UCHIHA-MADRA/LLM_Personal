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
  SafeAreaView,
  StatusBar,
  ActivityIndicator,
  Modal,
  ScrollView,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';

// ─── CONFIG ───
const getApiBase = () => {
  if (Platform.OS === 'web') {
    return typeof window !== 'undefined' && window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1'
      ? `http://${window.location.hostname}:8000`
      : 'http://127.0.0.1:8000';
  }
  return Platform.OS === 'android' ? 'http://10.0.2.2:8000' : 'http://127.0.0.1:8000';
};
const API_BASE = getApiBase();
const isWeb = Platform.OS === 'web';

type Role = 'user' | 'assistant';
type Message = { id: string; role: Role; content: string };
type ModelStatus = { loaded: boolean; name?: string; size_gb?: number };

export default function App() {
  const [messages, setMessages] = useState<Message[]>([
    { id: '1', role: 'assistant', content: 'Hello! I am your Personal LLM running on your desktop. Ask me anything!' },
  ]);
  const [input, setInput] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [status, setStatus] = useState<ModelStatus>({ loaded: false });
  const [catalog, setCatalog] = useState<any[]>([]);
  const [isModalVisible, setModalVisible] = useState(false);
  const [isLoadingModel, setIsLoadingModel] = useState(false);
  const flatListRef = useRef<FlatList>(null);

  // Check backend status and fetch catalog
  const fetchStatus = () => {
    fetch(`${API_BASE}/api/status`)
      .then(res => res.json())
      .then(data => setStatus(data))
      .catch(() => setStatus({ loaded: false }));
  };

  const fetchCatalog = () => {
    fetch(`${API_BASE}/api/models`)
      .then(res => res.json())
      .then(data => setCatalog(data.catalog || []))
      .catch(console.error);
  };

  useEffect(() => {
    fetchStatus();
    fetchCatalog();
  }, []);

  const handleLoadModel = async (filename: string) => {
    setIsLoadingModel(true);
    try {
      const res = await fetch(`${API_BASE}/api/models/load`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename }),
      });
      if (res.ok) {
        fetchStatus();
        setModalVisible(false);
      } else {
        alert('Failed to load model.');
      }
    } catch {
      alert('Network error connecting to backend.');
    } finally {
      setIsLoadingModel(false);
    }
  };

  const handleSend = async () => {
    if (!input.trim() || isGenerating) return;

    const userMsg: Message = { id: Date.now().toString(), role: 'user', content: input };
    const asstId = (Date.now() + 1).toString();

    setMessages(prev => [...prev, userMsg, { id: asstId, role: 'assistant', content: '' }]);
    setInput('');
    setIsGenerating(true);

    try {
      const response = await fetch(`${API_BASE}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: input, max_tokens: 1024, temperature: 0.7 }),
      });

      if (!response.ok) throw new Error('Stream failed');

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();

      while (reader) {
        const { value, done } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              if (data.type === 'token') {
                setMessages(prev =>
                  prev.map(msg =>
                    msg.id === asstId ? { ...msg, content: msg.content + data.content } : msg
                  )
                );
              } else if (data.type === 'done' || data.type === 'error') {
                break;
              }
            } catch {}
          }
        }
      }
    } catch (e) {
      setMessages(prev =>
        prev.map(msg =>
          msg.id === asstId ? { ...msg, content: '⚠️ Could not connect to the backend. Check your PC.' } : msg
        )
      );
    } finally {
      setIsGenerating(false);
    }
  };

  const renderMessage = ({ item }: { item: Message }) => {
    const isUser = item.role === 'user';
    return (
      <View style={[styles.msgRow, isUser ? styles.msgRowUser : styles.msgRowBot]}>
        {/* Avatar */}
        <View style={[styles.avatar, isUser ? styles.avatarUser : styles.avatarBot]}>
          <Text style={styles.avatarText}>{isUser ? '👤' : '🤖'}</Text>
        </View>
        {/* Bubble */}
        <View style={[styles.bubble, isUser ? styles.bubbleUser : styles.bubbleBot]}>
          {item.content ? (
            <Text style={[styles.msgText, isUser ? styles.msgTextUser : styles.msgTextBot]}>
              {item.content}
            </Text>
          ) : isGenerating ? (
            <ActivityIndicator size="small" color="#818cf8" />
          ) : null}
        </View>
      </View>
    );
  };

  return (
    <SafeAreaView style={styles.safe}>
      <StatusBar barStyle="light-content" backgroundColor="#0B0E14" />
      <LinearGradient colors={['#0B0E14', '#1a1040', '#0B0E14']} style={styles.container}>

        {/* Header */}
        <View style={styles.header}>
          <TouchableOpacity onPress={() => { setModalVisible(true); fetchCatalog(); }}>
            <Text style={styles.headerTitle}>🧠 Personal LLM <Text style={{fontSize: 14}}>(Models)</Text></Text>
          </TouchableOpacity>
          <View style={styles.statusChip}>
            <View style={[styles.statusDot, { backgroundColor: status.loaded ? '#22c55e' : '#ef4444' }]} />
            <Text style={styles.statusText}>
              {status.loaded ? status.name : 'Not Connected'}
            </Text>
          </View>
        </View>

        {/* Model Catalog Modal */}
        <Modal visible={isModalVisible} animationType="slide" transparent={true} onRequestClose={() => setModalVisible(false)}>
          <View style={styles.modalOverlay}>
            <View style={styles.modalContent}>
              <View style={styles.modalHeader}>
                <Text style={styles.modalTitle}>LLM Catalog ({catalog.length} Models)</Text>
                <TouchableOpacity onPress={() => setModalVisible(false)}>
                  <Text style={styles.closeModalText}>✕</Text>
                </TouchableOpacity>
              </View>
              <ScrollView style={styles.catalogScroll}>
                {catalog.filter(c => c.is_downloaded).map(model => (
                  <View key={model.key} style={styles.modelCardDownloaded}>
                    <Text style={styles.modelName}>{model.name}</Text>
                    <Text style={styles.modelDesc}>{model.tier ? `Tier ${model.tier}` : 'Local'} • {model.size_gb} GB</Text>
                    <TouchableOpacity 
                      style={[styles.loadBtn, isLoadingModel && styles.loadBtnDisabled]} 
                      onPress={() => handleLoadModel(model.filename)}
                      disabled={isLoadingModel}
                    >
                      <Text style={styles.loadBtnText}>{isLoadingModel ? 'Loading...' : 'Load Model'}</Text>
                    </TouchableOpacity>
                  </View>
                ))}
                <Text style={styles.sectionTitle}>Available to Download on PC</Text>
                {catalog.filter(c => !c.is_downloaded).map(model => (
                  <View key={model.key} style={styles.modelCardAvailable}>
                    <Text style={styles.modelName}>{model.name}</Text>
                    <Text style={styles.modelDesc} numberOfLines={2}>{model.description}</Text>
                    <Text style={styles.modelMeta}>Tier {model.tier} • {model.license} • {model.size_gb} GB</Text>
                    <Text style={styles.downloadHint}>Download via Desktop App</Text>
                  </View>
                ))}
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
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: '#0B0E14' },
  container: { 
    flex: 1, 
    width: '100%', 
    maxWidth: isWeb ? 800 : '100%', 
    alignSelf: 'center', 
    borderLeftWidth: isWeb ? 1 : 0, 
    borderRightWidth: isWeb ? 1 : 0, 
    borderColor: 'rgba(255,255,255,0.06)' 
  },

  // Header
  header: {
    paddingHorizontal: 20, paddingTop: 12, paddingBottom: 12,
    borderBottomWidth: 1, borderBottomColor: 'rgba(255,255,255,0.06)',
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
  },
  headerTitle: { color: '#fff', fontSize: 22, fontWeight: '800' },
  statusChip: {
    flexDirection: 'row', alignItems: 'center', backgroundColor: 'rgba(255,255,255,0.06)',
    paddingHorizontal: 10, paddingVertical: 5, borderRadius: 20,
  },
  statusDot: { width: 8, height: 8, borderRadius: 4, marginRight: 6 },
  statusText: { color: '#aaa', fontSize: 12 },

  // Chat
  chatList: { paddingHorizontal: 16, paddingVertical: 12 },
  msgRow: { flexDirection: 'row', marginBottom: 16, alignItems: 'flex-end' },
  msgRowUser: { flexDirection: 'row-reverse' },
  msgRowBot: { flexDirection: 'row' },

  avatar: {
    width: 36, height: 36, borderRadius: 12, alignItems: 'center', justifyContent: 'center',
    marginHorizontal: 8,
  },
  avatarUser: { backgroundColor: '#4f46e5' },
  avatarBot: { backgroundColor: '#1e2330', borderWidth: 1, borderColor: 'rgba(255,255,255,0.1)' },
  avatarText: { fontSize: 16 },

  bubble: { maxWidth: '75%', padding: 14, borderRadius: 18 },
  bubbleUser: {
    backgroundColor: '#4f46e5',
    borderBottomRightRadius: 4,
  },
  bubbleBot: {
    backgroundColor: 'rgba(30,35,48,0.8)',
    borderBottomLeftRadius: 4,
    borderWidth: 1, borderColor: 'rgba(255,255,255,0.06)',
  },
  msgText: { fontSize: 15, lineHeight: 22 },
  msgTextUser: { color: '#fff' },
  msgTextBot: { color: '#e2e8f0' },

  // Input
  inputBar: {
    flexDirection: 'row', alignItems: 'flex-end',
    paddingHorizontal: 12, paddingVertical: 10,
    borderTopWidth: 1, borderTopColor: 'rgba(255,255,255,0.06)',
    backgroundColor: 'rgba(11,14,20,0.9)',
  },
  textInput: {
    flex: 1, backgroundColor: '#151923', color: '#fff', borderRadius: 16,
    paddingHorizontal: 16, paddingVertical: 12, fontSize: 15,
    maxHeight: 120, borderWidth: 1, borderColor: 'rgba(255,255,255,0.08)',
  },
  sendBtn: {
    width: 44, height: 44, borderRadius: 14, backgroundColor: '#4f46e5',
    alignItems: 'center', justifyContent: 'center', marginLeft: 8,
  },
  sendBtnDisabled: { backgroundColor: '#333' },
  sendBtnText: { color: '#fff', fontSize: 18 },

  // Modal
  modalOverlay: { flex: 1, backgroundColor: 'rgba(0,0,0,0.7)', justifyContent: 'flex-end' },
  modalContent: { backgroundColor: '#0B0E14', borderTopLeftRadius: 24, borderTopRightRadius: 24, height: '85%', padding: 20, borderWidth: 1, borderColor: 'rgba(255,255,255,0.1)' },
  modalHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20, borderBottomWidth: 1, borderBottomColor: 'rgba(255,255,255,0.1)', paddingBottom: 15 },
  modalTitle: { color: '#fff', fontSize: 20, fontWeight: 'bold' },
  closeModalText: { color: '#aaa', fontSize: 20, padding: 5 },
  catalogScroll: { flex: 1 },
  sectionTitle: { color: '#8b5cf6', fontSize: 14, fontWeight: 'bold', textTransform: 'uppercase', marginTop: 24, marginBottom: 12, letterSpacing: 1 },
  
  modelCardDownloaded: { backgroundColor: 'rgba(79, 70, 229, 0.1)', borderWidth: 1, borderColor: 'rgba(79, 70, 229, 0.3)', borderRadius: 16, padding: 16, marginBottom: 12 },
  modelCardAvailable: { backgroundColor: 'rgba(255,255,255,0.05)', borderRadius: 16, padding: 16, marginBottom: 12 },
  modelName: { color: '#fff', fontSize: 16, fontWeight: 'bold', marginBottom: 4 },
  modelDesc: { color: '#aaa', fontSize: 13, lineHeight: 18, marginBottom: 8 },
  modelMeta: { color: '#6366f1', fontSize: 12, fontWeight: 'bold', marginBottom: 8 },
  
  loadBtn: { backgroundColor: '#4f46e5', paddingVertical: 10, borderRadius: 10, alignItems: 'center', marginTop: 8 },
  loadBtnDisabled: { opacity: 0.5 },
  loadBtnText: { color: '#fff', fontWeight: 'bold' },
  downloadHint: { color: '#555', fontSize: 12, fontStyle: 'italic', marginTop: 4 },
});
