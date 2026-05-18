# FawnCortex

> An AI conversation system powered by multi-agent collaboration, primarily addressing real-time dialogue challenges in cascaded architectures.

FawnCortex (Think like a fawn and grow up) is an agent application designed for conversational tasks. It adopts a pipeline architecture of **fast front-stage response + deep back-stage reasoning**, where specialized sub-agents (Chat Agent, Emotion Agent, Brain core, Reflection judge) collaborate to balance low-latency dialogue and high-quality inference by truncating and streaming the brain core's incremental thought process.

The project leverages the [AgentScope](https://github.com/agentscope-ai/agentscope) framework for foundational agent workflows, while also incorporating implementations for long-term memory, web search, and academic paper retrieval. For voice interaction, speech recognition (ASR) and synthesis (TTS) are currently implemented via API calls, so some adaptation may be needed when selecting models for this project.

Essentially, the project remains focused on context management during conversational interaction. Its methodological contribution lies in providing a reference for context pipelines and serving as a learning resource for dialogue-interactive projects. Discussions and sharing are welcome.

---

## Features

- **Agent Collaboration Architecture**
  - **ChatAgent**: Front-stage dialogue agent that quickly generates natural language responses for users
  - **EmotionAgent**: Expression and tone controller that drives avatar animations and voice styles
  - **BrainAgent**: Back-stage brain hub that performs deep reasoning and tool calling based on the ReAct pattern, and injects <u>incremental</u> thought processes into the ChatAgent
  - **ReflectionAgent**: Reflection judge that evaluates response quality each round to determine whether to trigger dialogue intervention
- **Asynchronous Orchestration Pipeline**
  - Front-stage runs ChatAgent + EmotionAgent in parallel, ensuring dialogue is decoupled into a single LLM call without any AgentLoop
  - Back-stage runs BrainAgent asynchronously, continuously injecting thought processes into ChatAgent's context based on time thresholds to assist in generating high-value content
  - OutputScheduler manages a priority queue for output components, orchestrating TTS voice broadcast and expression output
- **Interaction Enhancement**
  - Downstream of the EmotionAgent, the project includes VTube Studio live2d avatar control for reference and usage
- **Web UI Context Monitoring**
  - In addition to the main interaction page `/live`, the server includes a monitoring page `/index` for observing and analyzing dialogue context

---

## Quick Start

### Requirements

- Python >= 3.11
- (Optional) VTube Studio (if avatar integration is desired, but it needs customized live2d avatar)

### 1. Quick Launch

```bash
git clone <repo-url>
cd fawncortex
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables

The project heavily relies on APIs and global parameter configurations. You need to load API Keys via a `.env` file. An `.env` template is provided in the project root (no real keys are uploaded; please create your own):

```bash
cp .env.example .env
```

> The configuration supports assigning different LLM models to different agents for cost-effectiveness trade-offs.

### 3. Run the Project

#### Mode 1: Pure Text CLI (Recommended for learning this pipeline)

```bash
python chat_cli.py
# later, you can typing directly on the terminal
```

- Direct text-based conversation in the terminal **(※Recommended for studying context flow)**
- Fully demonstrates the complete information pipeline: front-stage response → back-stage reasoning → midway report → summary supplement
- Supports TTS voice broadcast and VTS avatar (if VTube Studio is connected)

#### Mode 2: Web Service

```bash
python server.py
```

Access:

- Monitoring dashboard: `http://localhost:8080/`
- Dialogue interaction page: `http://localhost:8080/live`

Web service supports:

- Text input and ASR voice input (frontend includes [silero_vad](https://github.com/snakers4/silero-vad) VAD)
- TTS audio real-time streaming playback
- Real-time visualization of agent internal events (thinking snapshots, expression updates, tool calls, etc.)

---

## Project Structure

```
.
├── chat_cli.py                   # text CLI (recommended for learning context flow)
├── server.py                     # Web service
├── web_scheduler.py              # Web core engine
├── config.py                     # Configuration file
├── requirements.txt              # Python dependencies
├── .env                          # Environment variables (API Keys, etc.)
│
├── fawncortex/
│   ├── agent/                    # Agent definitions
│   │   ├── chat_agent.py         # Front-stage dialogue agent
│   │   ├── emotion_agent.py      # Expression / Tone
│   │   ├── brain_agent.py        # Back-stage brain agent (ReAct + Tool Calling)
│   │   └── reflection_agent.py   # Reflection judge agent
│   ├── base/                     # Base components
│   │   ├── simple_agent.py       # Simple agent base class (wrapping AgentScope classes)
│   │   └── memory.py             # Long-term memory wrapper (mem0)
│   ├── pipeline/                 # Pipelines and scheduling
│   │   ├── front_stage_pipeline.py   # Front-stage parallel pipeline
│   │   ├── back_stage_midway.py      # Midway reports and summaries
│   │   ├── event_controller.py       # Event bus
│   │   └── output_scheduler.py       # TTS / VTS output scheduler
│   ├── tools/                    # Toolset (can be extended; refer to AgentScope)
│   │   ├── search_memory.py      # Memory retrieval / recording
│   │   ├── online_search.py      # Web search
│   │   ├── paper_search.py       # Paper search and reading
│   │   ├── get_current_time.py   # Get current time
│   │   └── weather.py            # Weather query
│   ├── components/               # External component adapters
│   │   ├── voice/
│   │   │   ├── tts.py            # TTS speech synthesis
│   │   │   └── asr.py            # ASR speech recognition
│   │   ├── body/
│   │   │   ├── vts_controller.py     # VTube Studio controller
│   │   │   └── emotion_animate.py    # Expression animation mapping
│   │   └── webui/static/         # Web frontend static files
│   └── logger/                   # Logging and latency tracking
│
├── data/                         # Runtime data (databases, cached audio)
└── logs/                         # Runtime logs
```

> **Getting Started Tips**:
> - To quickly understand the project pipeline and context flow of dialogue -> read `chat_cli.py`
> - To quickly run the conversational interaction application ->  run `python server.py`
> - `web_scheduler.py` is the core engine of the web service, **do not run it directly**

---

## Project Bottlenecks & TODOs

### Conversational Interaction Tasks

The project still discusses conversational interaction tasks based on a half-duplex, cascaded approach. End-to-end conversational task definition and modeling have not yet been addressed.

### Reflection Agent Bottleneck

The reflection agent does not function as a fully independent agent and lacks its own context management mechanism (it relies on the ChatAgent's dialogue history). Therefore, the reflection agent currently has no implementation for experience feedback (few-shot) or other performance improvements (model fine-tuning). In this project, it is implemented solely as a classifier, which should be noted.

Because the core intent of this project is low latency, reflective reasoning during dialogue interaction should not occupy the main dialogue chain. Adding an AgentLoop or feedback loop to sub-agents would significantly increase time-to-first-token (similarly, after VTS body actions are executed, there is no information pipeline for the agent to know it has performed an action). Thus, the reflection agent is the most critical yet weakest mechanism in the project's dialogue chain.

### Multi-Agent Limitations

This project does not directly discuss the **memory agent** or the **ASR+VAD+TTS pipeline**. The reason is that the BrainAgent already implements memory retrieval and storage within its AgentLoop, and this project leverages the [mem0](https://github.com/mem0ai/mem0) memory framework to quickly implement memory functionality. For half-duplex scenarios, the implementation of memory storage and the kernel design of AgentLoop can be referenced from other excellent projects; this project focuses solely on context management of dialogue information.

---

## Configuration Details

All configurations are centralized in `config.py` and can be overridden via environment variables. Key configuration items:

| Config Item | Description | Default |
|-------------|-------------|---------|
| `FAWNCORTEX_PORT` | Web service port | `8259` |
| `AGENT_NAME` / `USER_NAME` | Default names for agent and user | `AgentName` / `UserName` |
| `BRAIN_TIMEOUT` | Maximum thinking time for BrainAgent (seconds)<br />i.e., how long the dialogue tolerates the brain agent's reasoning per interaction | `300` |
| `BRAIN_CUT_TIME_DURATION` | Midway report truncation threshold (seconds)<br />i.e., after how many seconds of incremental reasoning the process is cut and sent to the ChatAgent to trigger dialogue | `5` |
| `STREAM` | Whether to enable LLM streaming output<br />Must be enabled; otherwise incremental reasoning cannot trigger dialogue | `True` |
| `LLM_ROLE_GENERATE_KWARGS` | Role-specific generation parameters (e.g., whether to enable thinking)<br />Should be configured based on the inference paradigm of the LLM model used. It is recommended to disable deep-thinking mode for the ChatAgent | Configured per agent role |

For more details, please refer to the comments in `config.py`.

---

## License

[MIT](LICENSE)

---

## Acknowledgments

We welcome scholars and experts to submit Issues for discussion!
