import { useEffect, useRef, useState } from "react";
import { sendMessage } from "./api.js";
import MarkdownReply from "./components/MarkdownReply.jsx";
import AnalysisBadges from "./components/AnalysisBadges.jsx";

const EXAMPLES = [
  "Rs 4,999 was debited twice for one order. Please refund urgently!",
  "Someone called pretending to be from my bank and now Rs 20,000 is gone from my UPI.",
  "Where is my order #A1023? It was supposed to arrive yesterday.",
  "I'm locked out after entering the wrong password 3 times.",
];

let nextId = 1;

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const bottomRef = useRef(null);

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
      const data = await sendMessage(message);
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
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  function handleKeyDown(event) {
    // Enter sends, Shift+Enter adds a new line.
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      submit(input);
    }
  }

  return (
    <div className="app">
      <header>
        <h1>
          Indigo<span>.ai</span>
        </h1>
        <p>Intent-based support assistant</p>
      </header>

      <main className="chat">
        {messages.length === 0 && (
          <div className="empty">
            <p>Try one of these:</p>
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
              <AnalysisBadges analysis={msg.analysis} />
              <MarkdownReply text={msg.text} />
              <div className="meta">
                {msg.meta.model} · {msg.meta.latency_ms} ms
                {msg.meta.completion_tokens != null && ` · ${msg.meta.completion_tokens} tokens`}
              </div>
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
          value={input}
          onChange={(event) => setInput(event.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Describe your issue..."
          rows={2}
          maxLength={8000}
          disabled={loading}
        />
        <button type="submit" disabled={loading || !input.trim()}>
          {loading ? "Sending..." : "Send"}
        </button>
      </form>
    </div>
  );
}
