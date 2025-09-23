"use client";

import { useState } from "react";

const InteractiveLegend = ({
  archetypes,
  colorScale,
  onArchetypeToggle,
  selectedArchetypes = [],
  onSelectAll,
  onDeselectAll,
  colorByLabel = "Categories",
  colorBy,
  onColorByChange,
  colorByOptions = [],
}) => {
  const [isExpanded, setIsExpanded] = useState(true);

  const handleArchetypeClick = (archetype) => {
    onArchetypeToggle(archetype);
  };

  const handleSelectAll = () => {
    onSelectAll();
  };

  const handleDeselectAll = () => {
    onDeselectAll();
  };

  const selectedCount = selectedArchetypes.length;
  const totalCount = archetypes.length;

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700 p-4">
      {/* Color Mapping Selector */}
      <div className="mb-4">
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
          Color by:
        </label>
        <select
          value={colorBy}
          onChange={(e) => onColorByChange(e.target.value)}
          className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          {colorByOptions.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      </div>

      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200">
          {colorByLabel}
        </h3>
        <div className="flex gap-2">
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="p-1 rounded hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
            aria-label={isExpanded ? "Collapse" : "Expand"}
          >
            <svg
              className={`w-4 h-4 transition-transform ${
                isExpanded ? "rotate-180" : ""
              }`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M19 9l-7 7-7-7"
              />
            </svg>
          </button>
        </div>
      </div>

      {/* Control Buttons */}
      <div className="flex gap-2 mb-3">
        <button
          onClick={handleSelectAll}
          className="px-3 py-1 text-xs bg-blue-500 text-white rounded hover:bg-blue-600 transition-colors"
        >
          Select All
        </button>
        <button
          onClick={handleDeselectAll}
          className="px-3 py-1 text-xs bg-gray-500 text-white rounded hover:bg-gray-600 transition-colors"
        >
          Deselect All
        </button>
      </div>

      {/* Legend Items */}
      <div
        className={`transition-all duration-300 overflow-hidden ${
          isExpanded ? "max-h-96" : "max-h-32"
        }`}
      >
        <div className="space-y-2">
          {archetypes.map((archetype, index) => {
            const isSelected = selectedArchetypes.includes(archetype);
            const color = colorScale(archetype);

            return (
              <div
                key={index}
                className={`flex items-center p-2 rounded-lg cursor-pointer transition-all duration-200 ${
                  isSelected
                    ? "bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800"
                    : "bg-gray-50 dark:bg-gray-700 hover:bg-gray-100 dark:hover:bg-gray-600"
                }`}
                onClick={() => handleArchetypeClick(archetype)}
              >
                {/* Color Circle */}
                <div
                  className="w-4 h-4 rounded-full mr-3 flex-shrink-0"
                  style={{ backgroundColor: color }}
                />

                {/* Archetype Name */}
                <span
                  className={`text-sm flex-1 ${
                    isSelected
                      ? "text-blue-800 dark:text-blue-200 font-medium"
                      : "text-gray-700 dark:text-gray-300"
                  }`}
                >
                  {archetype.length > 30
                    ? archetype.substring(0, 30) + "..."
                    : archetype}
                </span>

                {/* Selection Indicator */}
                <div
                  className={`w-5 h-5 rounded border-2 flex items-center justify-center ${
                    isSelected
                      ? "bg-blue-500 border-blue-500"
                      : "border-gray-300 dark:border-gray-600"
                  }`}
                >
                  {isSelected && (
                    <svg
                      className="w-3 h-3 text-white"
                      fill="currentColor"
                      viewBox="0 0 20 20"
                    >
                      <path
                        fillRule="evenodd"
                        d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                        clipRule="evenodd"
                      />
                    </svg>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Search/Filter */}
      {isExpanded && (
        <div className="mt-3 pt-3 border-t border-gray-200 dark:border-gray-600">
          <div className="text-xs text-gray-500 dark:text-gray-400">
            Click {colorByLabel.toLowerCase()} to show/hide them in the plot
          </div>
        </div>
      )}
    </div>
  );
};

export default InteractiveLegend;
