"use client";

const DetailPanel = ({ selectedPersona, onClose }) => {
  if (!selectedPersona) {
    return (
      <div className="w-full h-full flex items-center justify-center bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded-lg">
        <div className="text-center text-gray-500 dark:text-gray-400">
          <div className="text-4xl mb-4">👤</div>
          <p className="text-lg font-medium">
            Click on a point to view details
          </p>
          <p className="text-sm">
            Select any persona from the scatter plot to see their profile
          </p>
        </div>
      </div>
    );
  }

  const formatText = (text) => {
    if (!text) return "N/A";
    return text.length > 200 ? text.substring(0, 200) + "..." : text;
  };

  const sections = [
    {
      title: "Basic Information",
      fields: [
        { label: "Name", value: selectedPersona.name },
        { label: "Age", value: selectedPersona.age },
        { label: "Location", value: selectedPersona.location },
        { label: "Archetype", value: selectedPersona.archetype },
      ],
    },
    {
      title: "Categorization",
      fields: [
        {
          label: "Appearance Category",
          value: selectedPersona.appearance_category,
        },
        {
          label: "Behavior Category",
          value: selectedPersona.behavior_category,
        },
      ],
    },
    {
      title: "Memoir & Narrative",
      fields: [
        { label: "Memoir", value: selectedPersona.memoir },
        {
          label: "Memoir Narrative",
          value: formatText(selectedPersona.memoir_narrative),
        },
        {
          label: "Memoir Summary",
          value: formatText(selectedPersona.memoir_summary),
        },
        {
          label: "Archetype Description",
          value: formatText(selectedPersona.archetype_description),
        },
      ],
    },
    {
      title: "Psychological Profile",
      fields: [
        {
          label: "Presenting Problems",
          value: formatText(selectedPersona.presenting_problems),
        },
        { label: "Appearance", value: formatText(selectedPersona.appearance) },
        { label: "Behavior", value: formatText(selectedPersona.behavior) },
        {
          label: "Mood & Affect",
          value: formatText(selectedPersona.mood_affect),
        },
        { label: "Speech", value: formatText(selectedPersona.speech) },
        {
          label: "Thought Content",
          value: formatText(selectedPersona.thought_content),
        },
        {
          label: "Insight & Judgment",
          value: formatText(selectedPersona.insight_judgment),
        },
        { label: "Cognition", value: formatText(selectedPersona.cognition) },
      ],
    },
    {
      title: "History & Background",
      fields: [
        {
          label: "Medical/Developmental History",
          value: formatText(selectedPersona.medical_developmental_history),
        },
        {
          label: "Family History",
          value: formatText(selectedPersona.family_history),
        },
        {
          label: "Educational/Vocational History",
          value: formatText(selectedPersona.educational_vocational_history),
        },
      ],
    },
    {
      title: "Functioning",
      fields: [
        {
          label: "Emotional/Behavioral Functioning",
          value: formatText(selectedPersona.emotional_behavioral_functioning),
        },
        {
          label: "Social Functioning",
          value: formatText(selectedPersona.social_functioning),
        },
      ],
    },
    {
      title: "Summary",
      fields: [
        {
          label: "Psychological Profile Summary",
          value: formatText(selectedPersona.summary_of_psychological_profile),
        },
        {
          label: "Persona Narrative",
          value: formatText(selectedPersona.persona_narrative),
        },
      ],
    },
  ];

  return (
    <div className="w-full h-full bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-600 to-purple-600 text-white p-4 flex justify-between items-center">
        <div>
          <h2 className="text-xl font-bold">{selectedPersona.name}</h2>
          <p className="text-blue-100 text-sm">{selectedPersona.archetype}</p>
        </div>
        <button
          onClick={onClose}
          className="text-white hover:text-gray-200 transition-colors p-1 rounded-full hover:bg-white/20"
          aria-label="Close panel"
        >
          <svg
            className="w-6 h-6"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M6 18L18 6M6 6l12 12"
            />
          </svg>
        </button>
      </div>

      {/* Content */}
      <div className="h-full overflow-y-auto p-4 space-y-6">
        {sections.map((section, sectionIndex) => (
          <div key={sectionIndex} className="space-y-3">
            <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 border-b border-gray-200 dark:border-gray-600 pb-2">
              {section.title}
            </h3>
            <div className="space-y-2">
              {section.fields.map((field, fieldIndex) => (
                <div
                  key={fieldIndex}
                  className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3"
                >
                  <div className="text-sm font-medium text-gray-600 dark:text-gray-300 mb-1">
                    {field.label}
                  </div>
                  <div className="text-gray-800 dark:text-gray-200 text-sm leading-relaxed">
                    {field.value}
                  </div>
                </div>
              ))}
            </div>
          </div>
        ))}

        {/* Metadata */}
        <div className="bg-gray-100 dark:bg-gray-700 rounded-lg p-3 mt-6">
          <div className="text-xs text-gray-500 dark:text-gray-400">
            Index: {selectedPersona.index} | Version: {selectedPersona.version}
          </div>
        </div>
      </div>
    </div>
  );
};

export default DetailPanel;
