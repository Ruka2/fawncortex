/**
 * Deerberry Web UI 前端逻辑
 * ==========================
 * 职责：
 * 1. WebSocket 连接管理（自动重连）
 * 2. 消息渲染（微信风格气泡）
 * 3. 四个辅助 Agent 面板的数据更新
 * 4. TTS 音频播放（Web Audio API）
 * 5. 用户输入处理
 */

// ============================================================
// 全局状态
// ============================================================

const AppState = {
    ws: null,
    connected: false,
    reconnectTimer: null,
    currentRound: 0,
    chatHistory: [],      // ChatAgent 上下文历史
    emotionHistory: [],   // EmotionAgent 历史
    reflectionHistory: [],// ReflectionAgent 历史
    brainSnapshots: [],   // BrainAgent 快照历史
    brainRenderedIters: 0,// BrainAgent 已渲染的 iteration 数量（用于增量更新）
    chatContext: [],      // ChatAgent 当前 memory（来自后端推送）
    chatContextLength: 0, // ChatAgent 上下文长度
    chatLastResponseTime: 0,// ChatAgent 上次响应耗时
    chatReflectionAction: '-', // ChatAgent 最新 Reflection 判决
    currentAudio: null,   // 当前正在播放的音频元素
    audioQueue: [],       // 音频播放队列
    isPlayingAudio: false,
    brainStatus: 'idle',          // BrainAgent 当前状态（idle/thinking/acting）
    ttsBubbleQueue: [],           // 已调度给 TTS 的消息队列，按播放顺序排列
    roundTtsAudioDuration: 0,     // 本轮已播放的 TTS 音频总时长（秒）
};

// DOM 元素缓存
const DOM = {};

function cacheDOM() {
    DOM.chatMessages = document.getElementById('chat-messages');
    DOM.userInput = document.getElementById('user-input');
    DOM.sendBtn = document.getElementById('send-btn');
    DOM.connectionStatus = document.getElementById('connection-status');
    DOM.roundIndicator = document.getElementById('round-indicator');
    DOM.ttsPlayer = document.getElementById('tts-player');

    // BrainAgent
    DOM.brainReasoning = document.getElementById('brain-reasoning');
    DOM.brainToolCalls = document.getElementById('brain-tool-calls');
    DOM.brainIters = document.getElementById('brain-iters');
    DOM.brainElapsed = document.getElementById('brain-elapsed');
    DOM.brainTools = document.getElementById('brain-tools');
    DOM.brainToolCount = document.getElementById('brain-tool-count');
    DOM.brainStatus = document.getElementById('brain-status');

    // ChatAgent
    DOM.chatContextList = document.getElementById('chat-context-list');
    DOM.chatContextLength = document.getElementById('chat-context-length');
    DOM.chatResponseTime = document.getElementById('chat-response-time');
    DOM.chatReflection = document.getElementById('chat-reflection');

    // EmotionAgent
    DOM.emotionCurrent = document.getElementById('emotion-current');
    DOM.emotionHistory = document.getElementById('emotion-history');
    DOM.emotionRaw = document.getElementById('emotion-raw');
    DOM.emotionElapsed = document.getElementById('emotion-elapsed');
    DOM.emotionRound = document.getElementById('emotion-round');

    // ReflectionAgent
    DOM.reflectionList = document.getElementById('reflection-list');

    // 名称配置
    DOM.agentNameInput = document.getElementById('agent-name-input');
    DOM.userNameInput = document.getElementById('user-name-input');
    DOM.saveNamesBtn = document.getElementById('save-names-btn');
}

// ============================================================
// WebSocket 连接
// ============================================================

function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;

    AppState.ws = new WebSocket(wsUrl);

    AppState.ws.onopen = () => {
        console.log('[WS] 已连接');
        AppState.connected = true;
        updateConnectionStatus(true);
        clearTimeout(AppState.reconnectTimer);
    };

    AppState.ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            handleServerMessage(data);
        } catch (e) {
            console.error('[WS] 消息解析失败:', e);
        }
    };

    AppState.ws.onclose = () => {
        console.log('[WS] 连接断开');
        AppState.connected = false;
        updateConnectionStatus(false);
        scheduleReconnect();
    };

    AppState.ws.onerror = (err) => {
        console.error('[WS] 连接错误:', err);
    };
}

function updateConnectionStatus(connected) {
    if (!DOM.connectionStatus) return;
    DOM.connectionStatus.textContent = connected ? '已连接' : '未连接';
    DOM.connectionStatus.className = connected ? 'connected' : 'disconnected';
}

function scheduleReconnect() {
    if (AppState.reconnectTimer) clearTimeout(AppState.reconnectTimer);
    AppState.reconnectTimer = setTimeout(() => {
        console.log('[WS] 尝试重连...');
        connectWebSocket();
    }, 3000);
}

// ============================================================
// 消息路由
// ============================================================

function handleServerMessage(data) {
    const eventType = data.type;
    const payload = data.data || {};

    switch (eventType) {
        case 'round_start':
            handleRoundStart(payload);
            break;
        case 'round_end':
            handleRoundEnd(payload);
            break;
        case 'user_message':
            handleUserMessage(payload);
            break;
        case 'chat_message':
            handleChatMessage(payload);
            break;
        case 'midway_message':
            handleMidwayMessage(payload);
            break;
        case 'emotion_update':
            handleEmotionUpdate(payload);
            break;
        case 'brain_snapshot':
            handleBrainSnapshot(payload);
            break;
        case 'brain_summary':
            handleBrainSummary(payload);
            break;
        case 'reflection_judgment':
            handleReflectionJudgment(payload);
            break;
        case 'output_scheduled':
            handleOutputScheduled(payload);
            break;
        case 'tts_text':
            handleTTSText(payload);
            break;
        case 'tts_started':
            handleTTSStarted(payload);
            break;
        case 'tts_audio':
            handleTTSAudio(payload);
            break;
        case 'interrupt':
            handleInterrupt();
            break;
        case 'error':
            handleError(payload);
            break;
        case 'system':
            if (payload.status === 'reset') {
                handleReset();
            } else {
                const sysText = payload.text || payload.message;
                if (sysText) {
                    addSystemMessage(sysText);
                }
            }
            break;
        case 'chat_context':
            handleChatContext(payload);
            break;
        default:
            console.warn('[WS] 未知事件类型:', eventType);
    }
}

// ============================================================
// 事件处理器
// ============================================================

function handleRoundStart(data) {
    AppState.currentRound = data.round_id;
    DOM.roundIndicator.textContent = `Round ${data.round_id}`;
    // addSystemMessage(`🚀 第 ${data.round_id} 轮开始`);

    // 新一轮开始时，清空 BrainAgent 的上一轮显示
    AppState.brainSnapshots = [];  // 新增：防止内存泄漏
    AppState.brainRenderedIters = 0;
    if (DOM.brainReasoning) {
        DOM.brainReasoning.innerHTML = '';
    }
    if (DOM.brainToolCalls) {
        DOM.brainToolCalls.innerHTML = '';
    }

    // 清理跨轮的 TTS 气泡队列，防止残留
    AppState.ttsBubbleQueue = [];

    // 重置本轮 TTS 音频时长统计
    AppState.roundTtsAudioDuration = 0;
}

function handleRoundEnd(data) {
    // 显示可折叠的统计卡片（替代原来的简单系统消息）
    addRoundStatsCard(data);

    // Brain 回到 idle
    updateBrainStatus('idle');
}

function handleUserMessage(data) {
    addChatBubble('user', data.text, data.round_id);
    // 同步添加到 ChatAgent 上下文面板
    AppState.chatHistory.push({
        role: 'user',
        text: data.text,
        round_id: data.round_id,
        source: null,
    });
    renderChatContext();
}

function handleChatMessage(data) {
    const sourceTag = data.source || 'chat';
    // 主聊天框气泡不立即渲染，由 tts_started 事件触发（与后端 TTS 播放同步）

    // 添加到 ChatAgent 上下文面板（右侧不受影响）
    AppState.chatHistory.push({
        role: data.role || 'assistant',
        text: data.text,
        round_id: data.round_id,
        source: sourceTag,
        status: 'pending',
    });
    renderChatContext();
}

function handleMidwayMessage(data) {
    // 主聊天框气泡不立即渲染，由 tts_started 事件触发（与后端 TTS 播放同步）

    // 同时添加到 ChatAgent 上下文面板（右侧不受影响）
    AppState.chatHistory.push({
        role: 'assistant',
        text: data.text,
        round_id: data.round_id,
        source: 'midway',
        status: 'pending',
    });
    renderChatContext();
}

function handleEmotionUpdate(data) {
    AppState.emotionHistory.push({
        emotion: data.emotion,
        round_id: data.round_id,
        timestamp: new Date().toISOString(),
    });
    updateEmotionStats(data);
    renderEmotionHistory();
}

function handleBrainSnapshot(data) {
    const snapshot = data;
    AppState.brainSnapshots.push(snapshot);

    updateBrainStatus(snapshot.sub_status);

    // 统计信息
    if (DOM.brainIters) DOM.brainIters.textContent = snapshot.total_iters || 0;
    if (DOM.brainElapsed) DOM.brainElapsed.textContent = (snapshot.elapsed_sec || 0) + 's';
    if (DOM.brainTools) DOM.brainTools.textContent = snapshot.has_used_tools ? '是' : '否';

    // ── reasoning 区域：增量更新 ──
    if (DOM.brainReasoning) {
        let iterContainer = DOM.brainReasoning.querySelector('.brain-iterations');
        let streamContainer = DOM.brainReasoning.querySelector('.brain-streaming');
        if (!iterContainer) {
            iterContainer = document.createElement('div');
            iterContainer.className = 'brain-iterations';
            DOM.brainReasoning.appendChild(iterContainer);
        }
        if (!streamContainer) {
            streamContainer = document.createElement('div');
            streamContainer.className = 'brain-streaming';
            DOM.brainReasoning.appendChild(streamContainer);
        }

        // 1. 只追加「新增」的已完成 iterations（基于 brainRenderedIters 判断）
        if (snapshot.iterations && snapshot.iterations.length > AppState.brainRenderedIters) {
            for (let i = AppState.brainRenderedIters; i < snapshot.iterations.length; i++) {
                const it = snapshot.iterations[i];
                const div = document.createElement('div');
                div.className = 'brain-iteration';
                let content = `【第 ${it.iter} 轮】\n${it.reasoning_text || ''}`;
                if (it.acting && it.acting.tool_name) {
                    content += `\n→ 调用工具: ${it.acting.tool_name}`;
                }
                div.textContent = content;
                iterContainer.appendChild(div);
            }
            AppState.brainRenderedIters = snapshot.iterations.length;
        }

        // 2. 实时更新流式内容
        if (snapshot.stream_buffer && snapshot.stream_buffer.trim()) {
            streamContainer.textContent = `【思考中】\n${snapshot.stream_buffer}`;
        } else {
            streamContainer.textContent = '';
        }

        DOM.brainReasoning.scrollTop = DOM.brainReasoning.scrollHeight;
    }

    // ── tool calls 区域：每次全量清空重建 ──
    if (DOM.brainToolCalls && snapshot.iterations) {
        DOM.brainToolCalls.innerHTML = '';
        let validToolCount = 0;
        snapshot.iterations.forEach(it => {
            if (it.acting && it.acting.tool_name) {
                validToolCount++;
                const div = document.createElement('div');
                div.className = 'tool-call';
                let inputStr = '';
                try {
                    inputStr = JSON.stringify(it.acting.tool_input, null, 2);
                } catch (e) {
                    inputStr = String(it.acting.tool_input);
                }
                div.innerHTML = `
                    <div class="tool-name">🔧 ${it.acting.tool_name}</div>
                    <div class="tool-input">${escapeHtml(inputStr)}</div>
                `;
                DOM.brainToolCalls.appendChild(div);
            }
        });
        if (DOM.brainToolCount) {
            DOM.brainToolCount.textContent = validToolCount;
        }
    }
}

function handleBrainSummary(data) {
    // Brain 完成时推送最终快照
    if (data.snapshot) {
        handleBrainSnapshot({ ...data.snapshot, round_id: data.round_id });
    }
    // 可选：在 Brain 面板显示 summary 摘要
    // console.log('[Brain] Summary:', data.insight);
}

function handleReflectionJudgment(data) {
    AppState.reflectionHistory.push(data);
    renderReflectionHistory();

    // 标记被 Reflection 忽略的消息
    const ignoredActions = ['ignore', 'fatal_error'];
    if (ignoredActions.includes(data.action)) {
        // 优先精确匹配：round_id + agent_response 文本
        let item = AppState.chatHistory.findLast(
            m => m.round_id === data.round_id && m.text === data.agent_response
        );
        if (!item) {
            // 退而求其次：匹配来源和轮次
            item = AppState.chatHistory.findLast(
                m => m.round_id === data.round_id && m.source === data.source
            );
        }
        if (item) {
            item.status = 'ignored';
            renderChatContext();
        }
    }
}

function handleOutputScheduled(data) {
    // 消息已进入 TTS 队列，将其加入 ttsBubbleQueue，等待 playAudio 时同步渲染
    const roundId = AppState.currentRound;
    AppState.ttsBubbleQueue.push({
        text: data.text,
        source: data.source,
        round_id: roundId,
    });

    // 同时标记右侧 ChatAgent 面板中的消息为已播放
    let item = AppState.chatHistory.findLast(
        m => m.round_id === roundId && m.source === data.source && m.text === data.text
    );
    if (!item) {
        item = AppState.chatHistory.findLast(
            m => m.round_id === roundId && m.source === data.source
        );
    }
    if (item && item.status === 'pending') {
        item.status = 'played';
        renderChatContext();
    }
}

function handleTTSText(data) {
    // 可选：在前端显示即将播报的文本预览
    // console.log('[TTS] 即将播报:', data.text);
}

function handleTTSStarted(data) {
    // 后端 TTS 开始播放时，从队列取出对应消息并渲染气泡（与 TTS 同步）
    const bubbleData = AppState.ttsBubbleQueue.shift();
    if (bubbleData) {
        addChatBubble('assistant', bubbleData.text, bubbleData.round_id, bubbleData.source);
    } else {
        console.warn('[TTS] tts_started 触发但 ttsBubbleQueue 为空，可能 output_scheduled 未收到');
    }
}

function handleTTSAudio(data) {
    AppState.audioQueue.push(data);
    if (!AppState.isPlayingAudio) {
        processAudioQueue();
    }
}

function handleInterrupt() {
    const wasPlaying = AppState.isPlayingAudio;
    const wasThinking = AppState.brainStatus === 'thinking' || AppState.brainStatus === 'acting';
    if (wasPlaying || wasThinking) {
        addSystemMessage('🔇 用户打断，清空输出队列');
    }
    stopCurrentAudio();
    AppState.audioQueue = [];
    AppState.ttsBubbleQueue = [];
}

function handleReset() {
    // """处理服务器 reset 事件：清空前端所有 UI 状态。"""
    console.log('[UI] 收到 reset 事件，清空前端状态');

    // 1. 清空主聊天窗口
    if (DOM.chatMessages) {
        DOM.chatMessages.innerHTML = '';
    }

    // 2. 清空全局状态数组
    AppState.chatHistory = [];
    AppState.emotionHistory = [];
    AppState.reflectionHistory = [];
    AppState.brainSnapshots = [];
    AppState.brainRenderedIters = 0;
    AppState.chatContext = [];
    AppState.chatContextLength = 0;
    AppState.chatLastResponseTime = 0;
    AppState.chatReflectionAction = '-';
    AppState.ttsBubbleQueue = [];
    AppState.roundTtsAudioDuration = 0;
    AppState.currentRound = 0;

    // 3. 重置 BrainAgent 面板
    updateBrainStatus('idle');
    if (DOM.brainIters) DOM.brainIters.textContent = '0';
    if (DOM.brainElapsed) DOM.brainElapsed.textContent = '0s';
    if (DOM.brainTools) DOM.brainTools.textContent = '否';
    if (DOM.brainToolCount) DOM.brainToolCount.textContent = '0';
    if (DOM.brainReasoning) DOM.brainReasoning.innerHTML = '';
    if (DOM.brainToolCalls) DOM.brainToolCalls.innerHTML = '';

    // 4. 重置 ChatAgent 面板
    if (DOM.chatContextLength) DOM.chatContextLength.textContent = '0';
    if (DOM.chatResponseTime) DOM.chatResponseTime.textContent = '0s';
    if (DOM.chatReflection) DOM.chatReflection.textContent = '-';
    if (DOM.chatContextList) DOM.chatContextList.innerHTML = '';

    // 5. 重置 EmotionAgent 面板
    if (DOM.emotionRound) DOM.emotionRound.textContent = '-';
    if (DOM.emotionElapsed) DOM.emotionElapsed.textContent = '0s';
    if (DOM.emotionHistory) DOM.emotionHistory.innerHTML = '';

    // 6. 重置 ReflectionAgent 面板
    if (DOM.reflectionList) DOM.reflectionList.innerHTML = '';

    // 7. 重置 Round 指示器
    if (DOM.roundIndicator) DOM.roundIndicator.textContent = 'Round 0';

    // 8. 停止音频
    stopCurrentAudio();
    AppState.audioQueue = [];

    addSystemMessage('🗑️ 所有智能体短期记忆已清空');
}

function handleError(data) {
    addSystemMessage(`❌ 错误: ${data.message || '未知错误'}`);
}

function handleChatContext(data) {
    AppState.chatContext = data.messages || [];
    AppState.chatContextLength = data.context_length || 0;
    AppState.chatLastResponseTime = data.last_response_time || 0;
    AppState.chatReflectionAction = data.reflection_action || '-';
    renderChatContext();
}

// ============================================================
// UI 渲染函数
// ============================================================

function addChatBubble(role, text, roundId, sourceTag = null) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role}`;

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = role === 'user' ? '👤' : '🦌';

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    bubble.textContent = text;

    // 添加音频播放按钮（assistant 消息）
    if (role === 'assistant') {
        const meta = document.createElement('div');
        meta.className = 'message-meta';

        if (sourceTag) {
            const tag = document.createElement('span');
            tag.className = `source-tag ${sourceTag}`;
            tag.textContent = sourceTag;
            meta.appendChild(tag);
        }

        const roundSpan = document.createElement('span');
        roundSpan.textContent = `Round ${roundId}`;
        meta.appendChild(roundSpan);

        bubble.appendChild(meta);
    }

    msgDiv.appendChild(avatar);
    msgDiv.appendChild(bubble);

    DOM.chatMessages.appendChild(msgDiv);
    scrollToBottom();
}

function addSystemMessage(text) {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message system';

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    bubble.textContent = text;

    msgDiv.appendChild(bubble);
    DOM.chatMessages.appendChild(msgDiv);
    scrollToBottom();
}

function addRoundStatsCard(data) {
    // """渲染本轮统计卡片（可点击展开/折叠），显示性能评估指标。"""
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message system';

    const card = document.createElement('div');
    card.className = 'round-stats-card';

    // ── 头部（可点击折叠）──
    const header = document.createElement('div');
    header.className = 'round-stats-header';
    header.style.cursor = 'pointer';

    const titleSpan = document.createElement('span');
    titleSpan.className = 'round-stats-title';
    titleSpan.textContent = `⏱ 第 ${data.round_id} 轮统计`;

    const toggleSpan = document.createElement('span');
    toggleSpan.className = 'round-stats-toggle';
    toggleSpan.textContent = '▶';

    header.appendChild(titleSpan);
    header.appendChild(toggleSpan);

    // ── 折叠内容：详细指标 ──
    const detailsDiv = document.createElement('div');
    detailsDiv.className = 'round-stats-details';
    detailsDiv.style.display = 'none'; // 默认折叠

    const metrics = [
        {
            label: '首字延迟',
            desc: '用户输入 → 首次听到语音',
            value: `${data.user_perceived_s.toFixed(2)}s`,
        },
        {
            label: '语音总时长',
            desc: '本轮 TTS 音频累计播放时长',
            value: `${AppState.roundTtsAudioDuration.toFixed(2)}s`,
        },
        {
            label: '处理总时长',
            desc: '整轮处理消耗（不含 TTS）',
            value: `${data.round_without_tts_s.toFixed(2)}s`,
        },
    ];

    metrics.forEach(m => {
        const row = document.createElement('div');
        row.className = 'round-stats-row';

        const labelDiv = document.createElement('div');
        labelDiv.className = 'round-stats-label';
        labelDiv.textContent = m.label;

        const descDiv = document.createElement('div');
        descDiv.className = 'round-stats-desc';
        descDiv.textContent = m.desc;

        const valueDiv = document.createElement('div');
        valueDiv.className = 'round-stats-value';
        valueDiv.textContent = m.value;

        row.appendChild(labelDiv);
        row.appendChild(descDiv);
        row.appendChild(valueDiv);
        detailsDiv.appendChild(row);
    });

    // 点击头部折叠/展开
    header.addEventListener('click', () => {
        const isHidden = detailsDiv.style.display === 'none';
        detailsDiv.style.display = isHidden ? 'block' : 'none';
        toggleSpan.textContent = isHidden ? '▼' : '▶';
    });

    card.appendChild(header);
    card.appendChild(detailsDiv);
    msgDiv.appendChild(card);
    DOM.chatMessages.appendChild(msgDiv);
    scrollToBottom();
}

function scrollToBottom() {
    DOM.chatMessages.scrollTop = DOM.chatMessages.scrollHeight;
}

function updateBrainStatus(status) {
    AppState.brainStatus = status;
    if (!DOM.brainStatus) return;
    // 直接显示英文状态，不做中文映射
    DOM.brainStatus.textContent = status;
    DOM.brainStatus.className = `status-badge ${status}`;
}

function updateEmotionStats(data) {
    if (DOM.emotionCurrent) DOM.emotionCurrent.textContent = data.emotion || '-';
    if (DOM.emotionRaw) DOM.emotionRaw.textContent = data.raw || '-';
    if (DOM.emotionElapsed) DOM.emotionElapsed.textContent = (data.elapsed || 0) + 's';
    if (DOM.emotionRound) DOM.emotionRound.textContent = data.round_id || '-';
}

function renderChatContext() {
    if (!DOM.chatContextList) return;
    DOM.chatContextList.innerHTML = '';

    const currentRound = AppState.currentRound;
    const listDiv = document.createElement('div');
    listDiv.style.overflowY = 'auto';
    listDiv.style.flex = '1';

    AppState.chatHistory.forEach(item => {
        const div = document.createElement('div');
        const isSystem = item.role === 'user' && item.content && item.content.includes('[系统提示]');
        div.className = `context-item ${item.role} ${item.status || ''} ${isSystem ? 'system-prompt' : ''}`;

        const roleSpan = document.createElement('span');
        roleSpan.className = 'context-role';
        roleSpan.textContent = item.role;

        const textSpan = document.createElement('span');
        textSpan.className = 'context-text';
        // 系统提示灰显
        if (isSystem) {
            textSpan.style.color = '#999';
            textSpan.style.fontStyle = 'italic';
        }
        textSpan.textContent = item.text || '';

        div.appendChild(roleSpan);
        div.appendChild(textSpan);

        if (item.source) {
            const tag = document.createElement('span');
            tag.className = `source-tag ${item.source}`;
            tag.textContent = item.source;
            div.appendChild(tag);
        }

        listDiv.appendChild(div);
    });

    DOM.chatContextList.appendChild(listDiv);

    // 更新统计数字
    if (DOM.chatContextLength) DOM.chatContextLength.textContent = AppState.chatContextLength;
    if (DOM.chatResponseTime) DOM.chatResponseTime.textContent = (AppState.chatLastResponseTime || 0).toFixed(2) + 's';
    if (DOM.chatReflection) DOM.chatReflection.textContent = AppState.chatReflectionAction;
}

function renderEmotionHistory() {
    if (!DOM.emotionHistory) return;
    DOM.emotionHistory.innerHTML = '';

    // 栈式顺序：Round 大的在上（新的在上），类似堆栈
    const recent = AppState.emotionHistory.slice(-30).reverse();
    recent.forEach((item, idx) => {
        const div = document.createElement('div');
        const isLatest = idx === 0;  // reverse 后，索引 0 是最新的
        div.className = `emotion-item ${item.emotion} ${isLatest ? 'emotion-current-item' : ''}`;

        // 左侧：emotion 名称 + 当前表情标签（紧挨着）
        let leftHtml = `<span class="emotion-label">${item.emotion}</span>`;
        if (isLatest) {
            leftHtml += `<span class="emotion-current-tag">当前表情</span>`;
        }

        // 右侧：Round 序号
        const rightHtml = `<span class="emotion-round">Round ${item.round_id}</span>`;

        div.innerHTML = `<div class="emotion-left">${leftHtml}</div>${rightHtml}`;
        DOM.emotionHistory.appendChild(div);
    });
}

function renderReflectionHistory() {
    if (!DOM.reflectionList) return;
    DOM.reflectionList.innerHTML = '';

    const recent = AppState.reflectionHistory.slice(-20).reverse();
    recent.forEach((item, idx) => {
        const card = document.createElement('div');
        card.className = `reflection-item ${item.action}`;

        // ── 头部（可点击折叠）──
        const header = document.createElement('div');
        header.className = 'reflection-header';
        header.style.cursor = 'pointer';

        const actionSpan = document.createElement('span');
        actionSpan.className = 'reflection-action';
        actionSpan.textContent = item.action;

        const metaSpan = document.createElement('span');
        metaSpan.className = 'reflection-meta';
        metaSpan.textContent = `来源: ${item.source} | Round ${item.round_id}`;

        const toggleSpan = document.createElement('span');
        toggleSpan.className = 'reflection-toggle';
        toggleSpan.textContent = '▶';

        header.appendChild(actionSpan);
        header.appendChild(metaSpan);
        header.appendChild(toggleSpan);

        // ── 被审查的 agent 回复摘要 ──
        const summaryDiv = document.createElement('div');
        summaryDiv.className = 'reflection-summary';
        summaryDiv.textContent = item.agent_response;

        // ── 折叠内容：完整 chat_history ──
        const detailsDiv = document.createElement('div');
        detailsDiv.className = 'reflection-details';
        detailsDiv.style.display = 'none'; // 默认折叠

        if (item.chat_history && item.chat_history.length > 0) {
            const msgList = document.createElement('div');
            msgList.className = 'reflection-msg-list';
            item.chat_history.forEach(msg => {
                const msgDiv = document.createElement('div');
                msgDiv.className = `reflection-msg ${msg.role}`;
                const roleLabel = document.createElement('div');
                roleLabel.className = 'reflection-msg-role';
                roleLabel.textContent = `${msg.role} (${msg.name || ''})`;
                const contentDiv = document.createElement('div');
                contentDiv.className = 'reflection-msg-content';
                contentDiv.textContent = msg.content;
                msgDiv.appendChild(roleLabel);
                msgDiv.appendChild(contentDiv);
                msgList.appendChild(msgDiv);
            });
            detailsDiv.appendChild(msgList);
        } else {
            const emptyDiv = document.createElement('div');
            emptyDiv.className = 'reflection-empty';
            emptyDiv.textContent = '(无上下文)';
            detailsDiv.appendChild(emptyDiv);
        }

        // 点击头部折叠/展开
        header.addEventListener('click', () => {
            const isHidden = detailsDiv.style.display === 'none';
            detailsDiv.style.display = isHidden ? 'block' : 'none';
            toggleSpan.textContent = isHidden ? '▼' : '▶';
        });

        card.appendChild(header);
        card.appendChild(summaryDiv);
        card.appendChild(detailsDiv);
        DOM.reflectionList.appendChild(card);
    });
}

// ============================================================
// 音频播放
// ============================================================

function ensureAudioContext() {
    // 浏览器自动播放策略：首次用户交互后解锁 AudioContext
    if (window.audioContext && window.audioContext.state === 'suspended') {
        window.audioContext.resume().catch(() => {});
    }
}

function unlockAudioContext() {
    if (!window.audioContext) {
        window.audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (window.audioContext.state === 'suspended') {
        window.audioContext.resume().catch(() => {});
    }
}

function playAudio(audioData) {
    AppState.isPlayingAudio = true;

    // 【关键改动】与 TTS 播放同步渲染主聊天框气泡
    const bubbleData = AppState.ttsBubbleQueue.shift();
    if (bubbleData) {
        addChatBubble('assistant', bubbleData.text, bubbleData.round_id, bubbleData.source);
    }

    ensureAudioContext();

    // base64 → Blob → URL
    const byteCharacters = atob(audioData.audio_base64);
    const byteNumbers = new Array(byteCharacters.length);
    for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
    }
    const byteArray = new Uint8Array(byteNumbers);
    const blob = new Blob([byteArray], { type: audioData.mime_type || 'audio/mp3' });
    const url = URL.createObjectURL(blob);

    DOM.ttsPlayer.src = url;
    DOM.ttsPlayer.onloadedmetadata = () => {
        const dur = DOM.ttsPlayer.duration;
        if (dur && !isNaN(dur)) {
            AppState.roundTtsAudioDuration += dur;
        }
    };
    DOM.ttsPlayer.onended = () => {
        URL.revokeObjectURL(url);
        AppState.isPlayingAudio = false;
        processAudioQueue();
    };
    DOM.ttsPlayer.onerror = () => {
        console.error('[Audio] 播放失败');
        URL.revokeObjectURL(url);
        AppState.isPlayingAudio = false;
        processAudioQueue();
    };

    DOM.ttsPlayer.play().catch(err => {
        console.warn('[Audio] play() 被浏览器阻止（需要用户交互）:', err);
        // 显示一个静音提示，但不阻塞队列
        URL.revokeObjectURL(url);
        AppState.isPlayingAudio = false;
        processAudioQueue();
    });
}

function processAudioQueue() {
    if (AppState.audioQueue.length === 0) {
        AppState.isPlayingAudio = false;
        return;
    }
    const audioData = AppState.audioQueue.shift();
    playAudio(audioData);
}

function stopCurrentAudio() {
    if (DOM.ttsPlayer) {
        DOM.ttsPlayer.pause();
        DOM.ttsPlayer.currentTime = 0;
        DOM.ttsPlayer.src = '';
    }
    AppState.isPlayingAudio = false;
}

// ============================================================
// 用户输入
// ============================================================

function sendMessage() {
    const text = DOM.userInput.value.trim();
    if (!text || !AppState.connected) return;

    unlockAudioContext();

    AppState.ws.send(JSON.stringify({
        type: 'user_input',
        text: text,
    }));

    DOM.userInput.value = '';
}

// ============================================================
// 工具函数
// ============================================================

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ============================================================
// 初始化
// ============================================================

document.addEventListener('DOMContentLoaded', () => {
    cacheDOM();
    connectWebSocket();

    if (DOM.sendBtn) {
        DOM.sendBtn.addEventListener('click', sendMessage);
    }

    if (DOM.userInput) {
        DOM.userInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
    }

    // 清空聊天记录按钮
    const clearBtn = document.getElementById('clear-chat-btn');
    if (clearBtn) {
        clearBtn.addEventListener('click', () => {
            if (!AppState.connected) {
                alert('未连接到服务器');
                return;
            }
            if (confirm('确定要清空所有智能体的短期记忆和聊天记录吗？')) {
                AppState.ws.send(JSON.stringify({
                    type: 'reset',
                }));
            }
        });
    }

    // 名称配置保存按钮
    if (DOM.saveNamesBtn) {
        DOM.saveNamesBtn.addEventListener('click', () => {
            if (!AppState.connected) {
                alert('未连接到服务器');
                return;
            }
            const agentName = DOM.agentNameInput ? DOM.agentNameInput.value.trim() : '';
            const userName = DOM.userNameInput ? DOM.userNameInput.value.trim() : '';
            if (!agentName || !userName) {
                alert('名称不能为空');
                return;
            }
            AppState.ws.send(JSON.stringify({
                type: 'set_names',
                agent_name: agentName,
                user_name: userName,
            }));
        });
    }

    // Panel 折叠/展开
    document.querySelectorAll('.panel-header').forEach(header => {
        header.addEventListener('click', () => {
            const panel = header.closest('.panel');
            if (panel) {
                panel.classList.toggle('collapsed');
                const toggle = header.querySelector('.panel-toggle');
                if (toggle) {
                    toggle.textContent = panel.classList.contains('collapsed') ? '▶' : '▼';
                }
            }
        });
    });
});
