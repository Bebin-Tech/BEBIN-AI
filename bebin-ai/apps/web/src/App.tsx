import { FormEvent, KeyboardEvent, useEffect, useMemo, useRef, useState } from "react";

const apiBaseUrl = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";
const storageKey = "bebin-ai-conversations";
const tokenKey = "bebin-ai-token";

type Role = "user" | "assistant";

type ChatMessage = {
  id: string;
  role: Role;
  content: string;
  toolResults?: ToolResult[];
};

type ToolResult = {
  name: string;
  content: string;
  metadata: Record<string, unknown>;
};

type Conversation = {
  id: string;
  title: string;
  messages: ChatMessage[];
  createdAt: string;
};

type User = {
  id: string;
  email: string;
};

type ApiConversation = {
  id: string;
  title: string;
  created_at: string;
  messages: Array<{
    id: string;
    role: Role;
    content: string;
  }>;
};

type ResumeChatResponse = {
  answer: string;
  matches: Array<{
    candidate_name: string;
    filename: string;
    text: string;
    score: number;
  }>;
};

type GenerationSettings = {
  maxNewTokens: number;
  temperature: number;
  topK: number;
  topP: number;
  repetitionPenalty: number;
};

type BackendCheck = {
  ok: boolean;
  detail: string;
};

type BackendState = {
  live: boolean;
  ready: boolean;
  checks: Record<string, BackendCheck>;
  checkedAt?: string;
};

const defaultSettings: GenerationSettings = {
  maxNewTokens: 64,
  temperature: 0.8,
  topK: 50,
  topP: 0.95,
  repetitionPenalty: 1.1,
};

export function App() {
  const [authToken, setAuthToken] = useState(() => localStorage.getItem(tokenKey) ?? "");
  const [user, setUser] = useState<User | null>(null);
  const [authEmail, setAuthEmail] = useState("user@bebin.local");
  const [authPassword, setAuthPassword] = useState("password123");
  const [conversations, setConversations] = useState<Conversation[]>(() => loadConversations());
  const [activeId, setActiveId] = useState(() => conversations[0]?.id ?? "");
  const [input, setInput] = useState("");
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [resumeFiles, setResumeFiles] = useState<File[]>([]);
  const [resumeCandidate, setResumeCandidate] = useState("");
  const [resumeQuestion, setResumeQuestion] = useState("");
  const [resumeAnswer, setResumeAnswer] = useState("");
  const [settings, setSettings] = useState<GenerationSettings>(defaultSettings);
  const [isSending, setIsSending] = useState(false);
  const [status, setStatus] = useState("Ready");
  const [backendState, setBackendState] = useState<BackendState>({
    live: false,
    ready: false,
    checks: {},
  });
  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  const activeConversation = useMemo(
    () => conversations.find((conversation) => conversation.id === activeId) ?? conversations[0],
    [activeId, conversations],
  );

  useEffect(() => {
    if (conversations.length === 0) {
      const conversation = createConversation();
      setConversations([conversation]);
      setActiveId(conversation.id);
    }
  }, [conversations.length]);

  useEffect(() => {
    if (!authToken) {
      return;
    }

    void loadRemoteState();
  }, [authToken]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [activeConversation?.messages, isSending]);

  useEffect(() => {
    void refreshBackendState();
    const timer = window.setInterval(() => {
      void refreshBackendState();
    }, 15000);
    return () => window.clearInterval(timer);
  }, []);

  async function sendMessage(event?: FormEvent) {
    event?.preventDefault();
    const message = input.trim();
    if (!message || !activeConversation || isSending) {
      return;
    }
    if (!authToken) {
      setStatus("Sign in required");
      return;
    }

    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: withFileNote(message, selectedFiles),
    };
    const assistantMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: "assistant",
      content: "",
    };

    setInput("");
    setSelectedFiles([]);
    setIsSending(true);
    setStatus("Generating");
    if (selectedFiles.length > 0) {
      setStatus("Uploading documents");
      await uploadSelectedFiles(selectedFiles);
    }
    setStatus("Generating");
    updateConversation(activeConversation.id, (conversation) => ({
      ...conversation,
      title: conversation.messages.length === 0 ? titleFromMessage(message) : conversation.title,
      messages: [...conversation.messages, userMessage, assistantMessage],
    }));

    try {
      let done: { conversation_id?: string; tool_results?: ToolResult[] } | undefined;
      try {
        done = await streamChat(message, activeConversation.id, (partialText) => {
          updateMessage(activeConversation.id, assistantMessage.id, partialText);
        });
      } catch (error) {
        if (!isConversationNotFound(error)) {
          throw error;
        }
        updateMessage(activeConversation.id, assistantMessage.id, "");
        setStatus("Recovering chat");
        done = await streamChat(message, undefined, (partialText) => {
          updateMessage(activeConversation.id, assistantMessage.id, partialText);
        });
      }
      if (done?.conversation_id && done.conversation_id !== activeConversation.id) {
        const persistedConversationId = done.conversation_id;
        updateConversation(activeConversation.id, (conversation) => ({
          ...conversation,
          id: persistedConversationId,
        }));
        setActiveId(persistedConversationId);
      }
      if (done?.tool_results && done.tool_results.length > 0) {
        updateConversation(done.conversation_id ?? activeConversation.id, (conversation) => ({
          ...conversation,
          messages: conversation.messages.map((existing) =>
            existing.id === assistantMessage.id ? { ...existing, toolResults: done.tool_results } : existing,
          ),
        }));
      }
      setStatus("Ready");
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : "Unknown error";
      void refreshBackendState();
      updateMessage(
        activeConversation.id,
        assistantMessage.id,
        `Unable to reach the local model API.\n\n\`\`\`text\n${errorMessage}\n\`\`\``,
      );
      setStatus("Backend unavailable");
    } finally {
      setIsSending(false);
    }
  }

  async function streamChat(
    message: string,
    conversationId: string | undefined,
    onText: (text: string) => void,
  ): Promise<{ conversation_id?: string; tool_results?: ToolResult[] } | undefined> {
    const response = await fetch(`${apiBaseUrl}/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${authToken}` },
      body: JSON.stringify({
        message,
        ...(conversationId ? { conversation_id: conversationId } : {}),
        max_new_tokens: settings.maxNewTokens,
        temperature: settings.temperature,
        top_k: settings.topK,
        top_p: settings.topP,
        repetition_penalty: settings.repetitionPenalty,
        use_rag: true,
        use_tools: true,
        rag_top_k: 4,
      }),
    });

    if (!response.ok || !response.body) {
      throw new Error(`Request failed with status ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let donePayload: { conversation_id?: string } | undefined;

    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }

      buffer += decoder.decode(value, { stream: true });
      const events = buffer.split("\n\n");
      buffer = events.pop() ?? "";
      for (const eventText of events) {
        const payload = handleSseEvent(eventText, onText);
        donePayload = payload ?? donePayload;
      }
    }

    if (buffer.trim()) {
      const payload = handleSseEvent(buffer, onText);
      donePayload = payload ?? donePayload;
    }
    void loadRemoteConversations();
    return donePayload;
  }

  async function newChat() {
    const conversation = authToken ? await createRemoteConversation() : createConversation();
    setConversations((current) => [conversation, ...current]);
    setActiveId(conversation.id);
    setInput("");
  }

  async function deleteConversation(id: string) {
    if (authToken) {
      await fetch(`${apiBaseUrl}/conversations/${id}`, {
        method: "DELETE",
        headers: { Authorization: `Bearer ${authToken}` },
      });
    }
    setConversations((current) => {
      const next = current.filter((conversation) => conversation.id !== id);
      if (activeId === id) {
        setActiveId(next[0]?.id ?? "");
      }
      return next;
    });
  }

  function updateConversation(id: string, updater: (conversation: Conversation) => Conversation) {
    setConversations((current) =>
      current.map((conversation) => (conversation.id === id ? updater(conversation) : conversation)),
    );
  }

  function updateMessage(conversationId: string, messageId: string, content: string) {
    updateConversation(conversationId, (conversation) => ({
      ...conversation,
      messages: conversation.messages.map((message) =>
        message.id === messageId ? { ...message, content } : message,
      ),
    }));
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void sendMessage();
    }
  }

  async function authenticate(mode: "login" | "register") {
    setStatus(mode === "login" ? "Signing in" : "Creating account");
    const response = await fetch(`${apiBaseUrl}/auth/${mode}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email: authEmail, password: authPassword }),
    });
    if (!response.ok) {
      setStatus("Auth failed");
      return;
    }

    const body = (await response.json()) as { token: string; user: User };
    localStorage.setItem(tokenKey, body.token);
    setAuthToken(body.token);
    setUser(body.user);
    setStatus("Ready");
  }

  function logout() {
    localStorage.removeItem(tokenKey);
    setAuthToken("");
    setUser(null);
    setConversations([createConversation()]);
    setActiveId("");
    setStatus("Signed out");
  }

  async function loadRemoteState() {
    const me = await fetch(`${apiBaseUrl}/auth/me`, {
      headers: { Authorization: `Bearer ${authToken}` },
    });
    if (!me.ok) {
      logout();
      return;
    }
    setUser((await me.json()) as User);
    await loadRemoteConversations();
  }

  async function loadRemoteConversations() {
    const response = await fetch(`${apiBaseUrl}/conversations`, {
      headers: { Authorization: `Bearer ${authToken}` },
    });
    if (!response.ok) {
      return;
    }

    const remote = ((await response.json()) as ApiConversation[]).map(fromApiConversation);
    if (remote.length === 0) {
      const conversation = await createRemoteConversation();
      setConversations([conversation]);
      setActiveId(conversation.id);
      return;
    }
    setConversations(remote);
    setActiveId((current) => (remote.some((conversation) => conversation.id === current) ? current : remote[0].id));
  }

  async function createRemoteConversation(): Promise<Conversation> {
    const response = await fetch(`${apiBaseUrl}/conversations`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${authToken}` },
      body: JSON.stringify({ title: "New conversation" }),
    });
    if (!response.ok) {
      return createConversation();
    }
    return fromApiConversation((await response.json()) as ApiConversation);
  }

  async function uploadSelectedFiles(files: File[]) {
    for (const file of files) {
      const form = new FormData();
      form.append("file", file);
      const response = await fetch(`${apiBaseUrl}/documents`, {
        method: "POST",
        headers: { Authorization: `Bearer ${authToken}` },
        body: form,
      });
      if (!response.ok) {
        throw new Error(`Document upload failed for ${file.name}`);
      }
    }
  }

  async function uploadResumeFiles() {
    if (!authToken) {
      setStatus("Sign in required");
      return;
    }
    if (resumeFiles.length === 0) {
      setStatus("Select resume PDFs");
      return;
    }
    setStatus("Uploading resumes");
    const form = new FormData();
    for (const file of resumeFiles) {
      form.append("files", file);
    }
    const response = await fetch(`${apiBaseUrl}/resumes/bulk`, {
      method: "POST",
      headers: { Authorization: `Bearer ${authToken}` },
      body: form,
    });
    if (!response.ok) {
      setStatus("Resume upload failed");
      setResumeAnswer(await response.text());
      return;
    }
    const uploaded = (await response.json()) as Array<{ candidate_name: string }>;
    setResumeFiles([]);
    setResumeAnswer(`Indexed ${uploaded.length} resumes: ${uploaded.map((resume) => resume.candidate_name).join(", ")}`);
    setStatus("Ready");
  }

  async function askResumeBot() {
    if (!authToken) {
      setStatus("Sign in required");
      return;
    }
    if (!resumeQuestion.trim()) {
      setStatus("Ask resume question");
      return;
    }
    setStatus("Searching resumes");
    const response = await fetch(`${apiBaseUrl}/resumes/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${authToken}` },
      body: JSON.stringify({
        question: resumeQuestion,
        candidate_name: resumeCandidate.trim() || null,
        top_k: 5,
      }),
    });
    if (!response.ok) {
      setStatus("Resume search failed");
      setResumeAnswer(await response.text());
      return;
    }
    const body = (await response.json()) as ResumeChatResponse;
    setResumeAnswer(body.answer);
    setStatus("Ready");
  }

  async function refreshBackendState() {
    try {
      const live = await fetch(`${apiBaseUrl}/live`);
      if (!live.ok) {
        setBackendState({ live: false, ready: false, checks: {}, checkedAt: new Date().toISOString() });
        return;
      }

      const ready = await fetch(`${apiBaseUrl}/ready`);
      const readyBody = (await ready.json()) as {
        status: string;
        checks?: Record<string, BackendCheck>;
      };
      setBackendState({
        live: true,
        ready: ready.ok && readyBody.status === "ready",
        checks: readyBody.checks ?? {},
        checkedAt: new Date().toISOString(),
      });
    } catch {
      setBackendState({ live: false, ready: false, checks: {}, checkedAt: new Date().toISOString() });
    }
  }

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark">B</span>
          <button className="sidebar-icon" type="button" aria-label="Toggle sidebar">▯</button>
        </div>
        <button className="primary-action" type="button" onClick={() => void newChat()}>
          New chat
        </button>
        <div className="sidebar-links" aria-label="Tools">
          <button type="button">Plugins</button>
          <button type="button">Scheduled Tasks</button>
          <button type="button">Swarm</button>
          <button type="button">Slides</button>
          <button type="button">Deep Research</button>
          <button type="button">Websites</button>
          <button type="button">Docs</button>
          <button type="button">Sheets</button>
          <button type="button">Design</button>
          <button type="button">Bebin Work <span>Beta</span></button>
          <button type="button">Bebin Code</button>
        </div>
        <div className="sidebar-label">Recents</div>
        <nav className="conversation-list" aria-label="Conversations">
          {conversations.map((conversation) => (
            <button
              className={`conversation-item ${conversation.id === activeConversation?.id ? "active" : ""}`}
              key={conversation.id}
              type="button"
              onClick={() => setActiveId(conversation.id)}
            >
              <span>{conversation.title}</span>
              <small>{conversation.messages.length} messages</small>
            </button>
          ))}
        </nav>
        <form className="auth-card" onSubmit={(event) => event.preventDefault()}>
          <input
            aria-label="Email"
            value={authEmail}
            onChange={(event) => setAuthEmail(event.target.value)}
          />
          <input
            aria-label="Password"
            type="password"
            value={authPassword}
            onChange={(event) => setAuthPassword(event.target.value)}
          />
          <div className="auth-actions">
            <button type="button" onClick={() => void authenticate("login")}>Login</button>
            <button type="button" onClick={() => void authenticate("register")}>Register</button>
          </div>
          {user && <small>Signed in as {user.email}</small>}
        </form>
        <section className="resume-card">
          <strong>Resume RAG</strong>
          <label className="resume-upload">
            <input
              multiple
              accept="application/pdf,.pdf"
              type="file"
              onChange={(event) => setResumeFiles(Array.from(event.target.files ?? []))}
            />
            {resumeFiles.length > 0 ? `${resumeFiles.length} PDFs selected` : "Upload bulk PDFs"}
          </label>
          <input
            aria-label="Candidate name"
            placeholder="Candidate name"
            value={resumeCandidate}
            onChange={(event) => setResumeCandidate(event.target.value)}
          />
          <textarea
            aria-label="Resume question"
            placeholder="Ask about a candidate, skill, or role..."
            rows={2}
            value={resumeQuestion}
            onChange={(event) => setResumeQuestion(event.target.value)}
          />
          <div className="resume-actions">
            <button type="button" onClick={() => void uploadResumeFiles()}>Index</button>
            <button type="button" onClick={() => void askResumeBot()}>Ask</button>
          </div>
          {resumeAnswer && <pre>{resumeAnswer}</pre>}
        </section>
        <button
          className="secondary-action"
          type="button"
          onClick={() => activeConversation && void deleteConversation(activeConversation.id)}
        >
          Delete chat
        </button>
        {user && (
          <button className="secondary-action" type="button" onClick={logout}>
            Sign out
          </button>
        )}
        <div className="account-card">
          <span>{user?.email?.slice(0, 1).toUpperCase() ?? "B"}</span>
          <strong>{user?.email ?? "Bebin"}</strong>
        </div>
      </aside>

      <section className={`chat-panel ${activeConversation?.messages.length ? "has-messages" : "is-empty"}`}>
        <header className="topbar">
          <div className="title-block">
            <p className="eyebrow">Bebin AI</p>
            <h1>{activeConversation?.title ?? "New Chat"}</h1>
          </div>
          <div className="plan-pill">Local model</div>
          <BackendStatus status={status} backendState={backendState} onRefresh={refreshBackendState} />
        </header>

        <section className="messages" aria-live="polite">
          {activeConversation?.messages.length ? (
            activeConversation.messages.map((message) => <MessageBubble key={message.id} message={message} />)
          ) : (
            <div className="empty-state">
              <h2>BEBIN</h2>
              <p>
                Connected to your local checkpoint through FastAPI. The current smoke model is tiny,
                but this is the real end-to-end chat and document retrieval path.
              </p>
            </div>
          )}
          {isSending && <div className="thinking">Generating from local checkpoint...</div>}
          <div ref={messagesEndRef} />
        </section>

        <form className="composer" onSubmit={sendMessage}>
          <details className="controls">
            <summary>Generation controls</summary>
            <div className="settings-row">
              <NumberField
                label="Max"
                value={settings.maxNewTokens}
                min={1}
                max={512}
                step={1}
                onChange={(value) => setSettings((current) => ({ ...current, maxNewTokens: value }))}
              />
              <NumberField
                label="Temp"
                value={settings.temperature}
                min={0}
                max={2}
                step={0.1}
                onChange={(value) => setSettings((current) => ({ ...current, temperature: value }))}
              />
              <NumberField
                label="Top K"
                value={settings.topK}
                min={0}
                max={200}
                step={1}
                onChange={(value) => setSettings((current) => ({ ...current, topK: value }))}
              />
              <NumberField
                label="Top P"
                value={settings.topP}
                min={0.01}
                max={1}
                step={0.01}
                onChange={(value) => setSettings((current) => ({ ...current, topP: value }))}
              />
              <NumberField
                label="Repeat"
                value={settings.repetitionPenalty}
                min={0.1}
                max={3}
                step={0.1}
                onChange={(value) => setSettings((current) => ({ ...current, repetitionPenalty: value }))}
              />
            </div>
          </details>

          {selectedFiles.length > 0 && (
            <div className="file-row">
              {selectedFiles.map((file) => (
                <span key={`${file.name}-${file.size}`}>{file.name}</span>
              ))}
            </div>
          )}

          <div className="input-row">
            <label className="file-button" title="Attach files for later document workflows">
              <input
                multiple
                type="file"
                onChange={(event) => setSelectedFiles(Array.from(event.target.files ?? []))}
              />
              +
            </label>
            <textarea
              aria-label="Message"
              placeholder="Message Bebin AI..."
              rows={1}
              value={input}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={handleKeyDown}
            />
            <button className="send-button" disabled={isSending || !input.trim()} type="submit">
              ↑
            </button>
          </div>
          <div className="composer-chips" aria-label="Bebin tools">
            <span>Document Search</span>
            <span>Calculator</span>
            <span>RAG</span>
            <span>Tools</span>
            <span>Local SLM</span>
          </div>
        </form>
      </section>
    </main>
  );
}

function BackendStatus({
  status,
  backendState,
  onRefresh,
}: {
  status: string;
  backendState: BackendState;
  onRefresh: () => void;
}) {
  const stateLabel = backendState.ready ? status : backendState.live ? "Model not ready" : "Backend unavailable";
  const stateClass = backendState.ready ? "ready" : backendState.live ? "warning" : "offline";
  const checks = Object.entries(backendState.checks);

  return (
    <details className={`api-status ${stateClass}`}>
      <summary>
        <span className={`status-dot ${stateClass}`} />
        <span>{stateLabel}</span>
      </summary>
      <div className="status-panel">
        <div className="status-panel-header">
          <strong>{apiBaseUrl}</strong>
          <button type="button" onClick={onRefresh}>Refresh</button>
        </div>
        {checks.length > 0 ? (
          <dl>
            {checks.map(([name, check]) => (
              <div key={name}>
                <dt>{name}</dt>
                <dd className={check.ok ? "check-ok" : "check-failed"}>{check.detail}</dd>
              </div>
            ))}
          </dl>
        ) : (
          <p>Start the FastAPI backend on port 8000.</p>
        )}
        {backendState.checkedAt && (
          <small>Checked {new Date(backendState.checkedAt).toLocaleTimeString()}</small>
        )}
      </div>
    </details>
  );
}

function MessageBubble({ message }: { message: ChatMessage }) {
  return (
    <article className={`message ${message.role}`}>
      <div className="avatar">{message.role === "user" ? "You" : "B"}</div>
      <div className="message-content">
        {message.toolResults && message.toolResults.length > 0 && (
          <div className="tool-results">
            {message.toolResults.map((result, index) => (
              <details key={`${result.name}-${index}`}>
                <summary>{result.name}</summary>
                <pre>{result.content}</pre>
              </details>
            ))}
          </div>
        )}
        <RenderedContent content={message.content || " "} />
      </div>
    </article>
  );
}

function RenderedContent({ content }: { content: string }) {
  const parts = content.split(/```/g);
  return (
    <>
      {parts.map((part, index) =>
        index % 2 === 1 ? (
          <pre key={index}>
            <code>{stripCodeLanguage(part)}</code>
          </pre>
        ) : (
          <p key={index}>{part}</p>
        ),
      )}
    </>
  );
}

function NumberField({
  label,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
}) {
  return (
    <label className="number-field">
      <span>{label}</span>
      <input
        max={max}
        min={min}
        step={step}
        type="number"
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
  );
}

function handleSseEvent(
  eventText: string,
  onText: (text: string) => void,
): { conversation_id?: string; tool_results?: ToolResult[] } | undefined {
  const event = eventText
    .split("\n")
    .find((line) => line.startsWith("event:"))
    ?.replace("event:", "")
    .trim();
  const data = eventText
    .split("\n")
    .find((line) => line.startsWith("data:"))
    ?.replace("data:", "")
    .trim();

  if (!event || !data || data === "{}") {
    return undefined;
  }

  if (event === "error") {
    throw new Error(data);
  }

  const payload = JSON.parse(data) as {
    text?: string;
    response?: string;
    conversation_id?: string;
    tool_results?: ToolResult[];
  };
  if (event === "token" && payload.text !== undefined) {
    onText(payload.text);
  }
  if (event === "done" && payload.response !== undefined) {
    onText(payload.response);
    return { conversation_id: payload.conversation_id, tool_results: payload.tool_results };
  }
  return undefined;
}

function loadConversations(): Conversation[] {
  const saved = localStorage.getItem(storageKey);
  if (!saved) {
    return [createConversation()];
  }

  try {
    const parsed = JSON.parse(saved) as Conversation[];
    return Array.isArray(parsed) && parsed.length > 0 ? parsed : [createConversation()];
  } catch {
    return [createConversation()];
  }
}

function createConversation(): Conversation {
  return {
    id: crypto.randomUUID(),
    title: "New conversation",
    messages: [],
    createdAt: new Date().toISOString(),
  };
}

function titleFromMessage(message: string): string {
  return message.slice(0, 42) || "New conversation";
}

function isConversationNotFound(error: unknown): boolean {
  return error instanceof Error && error.message.toLowerCase().includes("conversation not found");
}

function withFileNote(message: string, files: File[]): string {
  if (files.length === 0) {
    return message;
  }

  const names = files.map((file) => file.name).join(", ");
  return `${message}\n\nAttached files selected in UI: ${names}`;
}

function stripCodeLanguage(code: string): string {
  return code.replace(/^\w+\n/, "").trim();
}

function fromApiConversation(conversation: ApiConversation): Conversation {
  return {
    id: conversation.id,
    title: conversation.title,
    createdAt: conversation.created_at,
    messages: conversation.messages.map((message) => ({
      id: message.id,
      role: message.role,
      content: message.content,
    })),
  };
}
