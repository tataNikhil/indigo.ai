const pretty = (value) => value.replaceAll("_", " ");

export default function AnalysisBadges({ analysis }) {
  const { intent, confidence, risk, priority } = analysis;

  return (
    <div className="badges">
      <span className="badge">{pretty(intent)}</span>
      <span className={`badge risk-${risk.level}`}>risk: {risk.level}</span>
      <span className={`badge priority-${priority.level}`}>
        {priority.level} · {priority.sla_hours}h
      </span>
      {priority.requires_human && <span className="badge human">agent follow-up</span>}
      {risk.flags.map((flag) => (
        <span key={flag} className="badge flag">
          {pretty(flag)}
        </span>
      ))}
      <span className="confidence">{Math.round(confidence * 100)}% confident</span>
    </div>
  );
}
