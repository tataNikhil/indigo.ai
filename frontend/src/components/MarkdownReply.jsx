import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

// remark-gfm adds tables, strikethrough and task lists.
// react-markdown does not render raw HTML by default, so model output
// can't inject <script> tags into the page.
const components = {
  a: ({ node, ...props }) => <a {...props} target="_blank" rel="noopener noreferrer" />,
};

export default function MarkdownReply({ text }) {
  return (
    <div className="markdown">
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
        {text}
      </ReactMarkdown>
    </div>
  );
}
