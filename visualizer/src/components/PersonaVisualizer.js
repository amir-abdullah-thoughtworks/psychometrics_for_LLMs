"use client";

import { useState, useEffect } from "react";
import ScatterPlot from "./ScatterPlot";
import DetailPanel from "./DetailPanel";
import InteractiveLegend from "./InteractiveLegend";

const PersonaVisualizer = ({ personasData }) => {
  const [embeddingType, setEmbeddingType] = useState("pca");
  const [selectedPersona, setSelectedPersona] = useState(null);
  const [selectedArchetypes, setSelectedArchetypes] = useState([]);
  const [colorBy, setColorBy] = useState("archetype");
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    if (personasData) {
      setIsLoading(false);
    }
  }, [personasData]);

  const handlePointClick = (persona) => {
    setSelectedPersona(persona);
  };

  const handleClosePanel = () => {
    setSelectedPersona(null);
  };

  const handleEmbeddingToggle = (type) => {
    setEmbeddingType(type);
    setSelectedPersona(null); // Clear selection when switching views
  };

  const handleArchetypeToggle = (archetype) => {
    setSelectedArchetypes((prev) => {
      if (prev.includes(archetype)) {
        return prev.filter((a) => a !== archetype);
      } else {
        return [...prev, archetype];
      }
    });
  };

  const handleSelectAllArchetypes = () => {
    const allValues = getUniqueValues();
    setSelectedArchetypes(allValues);
  };

  const handleDeselectAllArchetypes = () => {
    setSelectedArchetypes([]);
  };

  const handleColorByChange = (newColorBy) => {
    setColorBy(newColorBy);
    setSelectedArchetypes([]); // Reset selections when changing color mapping
  };

  // Helper function to get age bin for any age value
  const getAgeBin = (age) => {
    const ageNum = parseInt(age);
    if (isNaN(ageNum)) return null;

    if (ageNum >= 21 && ageNum <= 25) return "21-25";
    if (ageNum >= 26 && ageNum <= 30) return "26-30";
    if (ageNum >= 31 && ageNum <= 35) return "31-35";
    if (ageNum >= 36 && ageNum <= 40) return "36-40";
    if (ageNum >= 41 && ageNum <= 45) return "41-45";
    if (ageNum >= 46 && ageNum <= 50) return "46-50";
    if (ageNum >= 51 && ageNum <= 55) return "51-55";
    if (ageNum >= 56 && ageNum <= 60) return "56-60";
    if (ageNum >= 61 && ageNum <= 65) return "61-65";
    if (ageNum >= 66 && ageNum <= 69) return "66-69";

    if (ageNum < 21) return "Under 21";
    if (ageNum > 69) return "Over 69";

    return null;
  };

  // Get unique values for the current color mapping
  const getUniqueValues = () => {
    if (!personasData || !personasData.data) return [];

    let values;
    if (colorBy === "age") {
      // Create age bins
      values = personasData.data
        .map((d) => {
          const age = parseInt(d.age);
          if (isNaN(age)) return null;

          if (age >= 21 && age <= 25) return "21-25";
          if (age >= 26 && age <= 30) return "26-30";
          if (age >= 31 && age <= 35) return "31-35";
          if (age >= 36 && age <= 40) return "36-40";
          if (age >= 41 && age <= 45) return "41-45";
          if (age >= 46 && age <= 50) return "46-50";
          if (age >= 51 && age <= 55) return "51-55";
          if (age >= 56 && age <= 60) return "56-60";
          if (age >= 61 && age <= 65) return "61-65";
          if (age >= 66 && age <= 69) return "66-69";

          // Handle ages outside the specified range
          if (age < 21) return "Under 21";
          if (age > 69) return "Over 69";

          return null;
        })
        .filter(Boolean);
    } else {
      values = personasData.data.map((d) => d[colorBy]).filter(Boolean);
    }

    return [...new Set(values)].sort();
  };

  // Create color scale function
  const createColorScale = () => {
    const customColors = [
      "#1f77b4",
      "#ff7f0e",
      "#2ca02c",
      "#d62728",
      "#9467bd",
      "#8c564b",
      "#e377c2",
      "#7f7f7f",
      "#bcbd22",
      "#17becf",
      "#aec7e8",
      "#ffbb78",
      "#98df8a",
      "#ff9896",
      "#c5b0d5",
      "#c49c94",
      "#f7b6d3",
      "#c7c7c7",
      "#dbdb8d",
      "#9edae5",
      "#393b79",
      "#5254a3",
      "#6b6ecf",
      "#9c9ede",
      "#637939",
      "#8ca252",
      "#b5cf6b",
      "#cedb9c",
      "#8c6d31",
      "#bd9e39",
    ];

    const uniqueValues = getUniqueValues();
    return (value) => {
      const index = uniqueValues.indexOf(value);
      return customColors[index % customColors.length];
    };
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600 dark:text-gray-400">
            Loading persona data...
          </p>
        </div>
      </div>
    );
  }

  if (!personasData) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-center">
          <div className="text-red-500 text-4xl mb-4">⚠️</div>
          <p className="text-gray-600 dark:text-gray-400">
            No persona data available
          </p>
        </div>
      </div>
    );
  }

  const { metadata, embeddings, data } = personasData;

  return (
    <div className="h-screen bg-gray-50 dark:bg-gray-900 flex flex-col">
      {/* Header */}
      <div className="bg-white dark:bg-gray-800 shadow-sm border-b border-gray-200 dark:border-gray-700 p-4">
        <div className="flex justify-between items-center">
          <div>
            <h1 className="text-2xl font-bold text-gray-800 dark:text-gray-200">
              Personas visualizer
            </h1>
            <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
              {metadata.total_points} personas •{" "}
              {metadata.original_embedding_dim}D → 2D embeddings
            </p>
          </div>

          {/* Embedding Type Toggle */}
          <div className="flex bg-gray-100 dark:bg-gray-700 rounded-lg p-1">
            <button
              onClick={() => handleEmbeddingToggle("pca")}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                embeddingType === "pca"
                  ? "bg-white dark:bg-gray-600 text-gray-900 dark:text-gray-100 shadow-sm"
                  : "text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200"
              }`}
            >
              PCA
            </button>
            <button
              onClick={() => handleEmbeddingToggle("umap")}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                embeddingType === "umap"
                  ? "bg-white dark:bg-gray-600 text-gray-900 dark:text-gray-100 shadow-sm"
                  : "text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200"
              }`}
            >
              UMAP
            </button>
          </div>
        </div>

        {/* Embedding Info */}
        <div className="mt-3 flex gap-6 text-sm text-gray-600 dark:text-gray-400">
          {embeddingType === "pca" && (
            <>
              <span>
                Variance Explained:{" "}
                {(metadata.pca_total_variance_explained * 100).toFixed(1)}%
              </span>
              <span>
                PC1:{" "}
                {(metadata.pca_explained_variance_ratio[0] * 100).toFixed(1)}%
              </span>
              <span>
                PC2:{" "}
                {(metadata.pca_explained_variance_ratio[1] * 100).toFixed(1)}%
              </span>
            </>
          )}
          {embeddingType === "umap" && (
            <>
              <span>Neighbors: {metadata.umap_parameters.n_neighbors}</span>
              <span>Min Distance: {metadata.umap_parameters.min_dist}</span>
              <span>Metric: {metadata.umap_parameters.metric}</span>
            </>
          )}
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col lg:flex-row overflow-hidden">
        {/* Left Side - Scatter Plot and Legend */}
        <div className="flex-1 flex flex-col lg:flex-row p-2 lg:p-4 gap-4">
          {/* Scatter Plot */}
          <div className="flex-1 bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-2 lg:p-4">
            <ScatterPlot
              data={data}
              embeddings={embeddings}
              embeddingType={embeddingType}
              onPointClick={handlePointClick}
              selectedPoint={selectedPersona}
              selectedArchetypes={selectedArchetypes}
              colorBy={colorBy}
              colorScale={createColorScale()}
              getAgeBin={getAgeBin}
              width={800}
              height={600}
            />
          </div>

          {/* Interactive Legend */}
          <div className="w-full lg:w-80">
            <InteractiveLegend
              archetypes={getUniqueValues()}
              colorScale={createColorScale()}
              onArchetypeToggle={handleArchetypeToggle}
              selectedArchetypes={selectedArchetypes}
              onSelectAll={handleSelectAllArchetypes}
              onDeselectAll={handleDeselectAllArchetypes}
              colorByLabel={colorBy
                .replace("_", " ")
                .replace(/\b\w/g, (l) => l.toUpperCase())}
              colorBy={colorBy}
              onColorByChange={handleColorByChange}
              colorByOptions={[
                { value: "archetype", label: "Archetype" },
                { value: "age", label: "Age" },
                { value: "sex", label: "Sex" },
                { value: "education_level", label: "Education Level" },
                { value: "bachelors_field", label: "Bachelor's Field" },
                { value: "ethnic_background", label: "Ethnic Background" },
                { value: "marital_status", label: "Marital Status" },
                { value: "appearance_category", label: "Appearance Category" },
                { value: "behavior_category", label: "Behavior Category" },
              ]}
            />
          </div>
        </div>

        {/* Detail Panel */}
        <div className="w-full lg:w-96 p-2 lg:p-4 lg:border-l border-gray-200 dark:border-gray-700">
          <DetailPanel
            selectedPersona={selectedPersona}
            onClose={handleClosePanel}
          />
        </div>
      </div>

      {/* Footer Stats */}
      <div className="bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700 p-3">
        <div className="flex justify-between text-sm text-gray-600 dark:text-gray-400">
          <span>
            {selectedArchetypes.length === 0
              ? `Showing all ${colorBy.replace("_", " ")}s`
              : `Filtered: ${selectedArchetypes.length} ${colorBy.replace(
                  "_",
                  " "
                )}s selected`}
          </span>
          <span>
            {selectedPersona
              ? `Selected: ${selectedPersona.name}`
              : "Click any point to view details"}
          </span>
        </div>
      </div>
    </div>
  );
};

export default PersonaVisualizer;
