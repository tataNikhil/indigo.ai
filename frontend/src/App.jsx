import { useEffect, useRef, useState } from "react";
import { deleteSession, getSession, sendMessage } from "./api.js";
import MarkdownReply from "./components/MarkdownReply.jsx";
import AnalysisBadges from "./components/AnalysisBadges.jsx";

const EXAMPLES = [
  "Rs 4,999 was debited twice for one order. Please refund urgently!",
  "Someone called pretending to be from my bank and now Rs 20,000 is gone from my UPI.",
  "Where is my order #A1023? It was supposed to arrive yesterday.",
  "I'm locked out after entering the wrong password 3 times.",
];

const SESSION_KEY = "indigo.sessionId";

function loadSavedSessionId() {
  try {
    return localStorage.getItem(SESSION_KEY);
  } catch {
    return null;
  }
}

function saveSessionId(id) {
  try {
    if (id) localStorage.setItem(SESSION_KEY, id);
    else localStorage.removeItem(SESSION_KEY);
  } catch {
    // ignore
  }
}

let nextId = 1;

function fromServer(message) {
  return {
    id: nextId++,
    role: message.role,
    text: message.content,
    analysis: message.analysis,
    meta: message.reply_metadata,
  };
}

export default function App() {
  const [sessionId, setSessionId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [restoring, setRestoring] = useState(true);
  const [error, setError] = useState("");
  const [slow, setSlow] = useState(false);
  const bottomRef = useRef(null);
  const inputRef = useRef(null);

  // free Render instances can take a while to wake up
  useEffect(() => {
    if (!loading && !restoring) {
      setSlow(false);
      return;
    }
    const timer = setTimeout(() => setSlow(true), 6000);
    return () => clearTimeout(timer);
  }, [loading, restoring]);

  function rememberSession(id) {
    setSessionId(id);
    saveSessionId(id);
  }

  useEffect(() => {
    const saved = loadSavedSessionId();
    if (!saved) {
      setRestoring(false);
      return;
    }
    getSession(saved)
      .then((session) => {
        setSessionId(session.session_id);
        setMessages(session.messages.map(fromServer));
      })
      .catch((err) => {
        if (err.status === 404) saveSessionId(null);
        else setError(err.message);
      })
      .finally(() => setRestoring(false));
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  async function submit(text) {
    const message = text.trim();
    if (!message || loading) return;

    setMessages((prev) => [...prev, { id: nextId++, role: "user", text: message }]);
    setInput("");
    setError("");
    setLoading(true);

    try {
      const data = await sendMessage(message, sessionId);
      rememberSession(data.session_id);
      setMessages((prev) => [
        ...prev,
        {
          id: nextId++,
          role: "assistant",
          text: data.reply_markdown,
          analysis: data.analysis,
          meta: data.reply_metadata,
        },
      ]);
    } catch (err) {
      if (err.status === 404) {
        rememberSession(null);
        setError("This conversation has expired. Send your message again to start a new chat.");
      } else {
        setError(err.message);
      }
    } finally {
      setLoading(false);
    }
  }

  async function newChat() {
    if (loading) return;
    const oldId = sessionId;
    rememberSession(null);
    setMessages([]);
    setError("");
    setInput("");
    inputRef.current?.focus();
    if (oldId) {
      deleteSession(oldId).catch(() => {});
    }
  }

  function handleKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      submit(input);
    }
  }

  return (
    <div className="app">
      <header>
        <div>
          <h1>
            Indigo<span>.ai</span>
          </h1>
          <p>
            Intent-based support assistant
            {sessionId && (
              <span className="session" title={sessionId}>
                {" "}
                · session {sessionId.slice(0, 8)}
              </span>
            )}
          </p>
        </div>
        <button className="new-chat" onClick={newChat} disabled={loading || messages.length === 0}>
          + New chat
        </button>
      </header>

      <main className="chat">
        {restoring && <p className="hint">Loading your conversation...</p>}

        {!restoring && messages.length === 0 && (
          <div className="empty">
            <p>Try one of these, then ask a follow-up like "how long will that take?"</p>
            {EXAMPLES.map((example) => (
              <button key={example} className="example" onClick={() => submit(example)}>
                {example}
              </button>
            ))}
          </div>
        )}

        {messages.map((msg) =>
          msg.role === "user" ? (
            <div key={msg.id} className="bubble user">
              {msg.text}
            </div>
          ) : (
            <div key={msg.id} className="bubble assistant">
              {msg.analysis && <AnalysisBadges analysis={msg.analysis} />}
              <MarkdownReply text={msg.text} />
              {msg.meta && (
                <div className="meta">
                  {msg.meta.model} · {msg.meta.latency_ms} ms
                  {msg.meta.completion_tokens != null && ` · ${msg.meta.completion_tokens} tokens`}
                  {msg.meta.history_messages_used > 0 &&
                    ` · remembered ${msg.meta.history_messages_used} earlier messages`}
                </div>
              )}
            </div>
          )
        )}

        {loading && (
          <div className="bubble assistant loading">
            <span className="dot" />
            <span className="dot" />
            <span className="dot" />
          </div>
        )}

        {slow && (
          <p className="hint">
            Waking up the server - the free hosting plan sleeps when idle, so the first request
            can take up to a minute.
          </p>
        )}

        {error && (
          <div className="error" role="alert">
            {error}
          </div>
        )}

        <div ref={bottomRef} />
      </main>

      <form
        className="composer"
        onSubmit={(event) => {
          event.preventDefault();
          submit(input);
        }}
      >
        <textarea
          ref={inputRef}
          value={input}
          onChange={(event) => setInput(event.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={messages.length ? "Ask a follow-up..." : "Describe your issue..."}
          rows={2}
          maxLength={8000}
          disabled={loading || restoring}
        />
        <button type="submit" disabled={loading || restoring || !input.trim()}>
          {loading ? "Sending..." : "Send"}
        </button>
      </form>
    </div>
  );
}
