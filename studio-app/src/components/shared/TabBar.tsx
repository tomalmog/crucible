interface TabBarProps<T extends string> {
  tabs: readonly T[];
  active: T;
  onChange: (tab: T) => void;
  format?: (tab: T) => string;
}

export function TabBar<T extends string>({ tabs, active, onChange, format }: TabBarProps<T>) {
  const fmt = format ?? ((t: T) => t.charAt(0).toUpperCase() + t.slice(1));
  return (
    <div className="tab-list">
      {tabs.map((t) => (
        <button
          key={t}
          className={`tab-item${t === active ? " active" : ""}`}
          onClick={() => onChange(t)}
        >
          {fmt(t)}
        </button>
      ))}
    </div>
  );
}
