"use client";

import { useEffect, useRef, useState } from "react";
import * as d3 from "d3";

const ScatterPlot = ({
  data,
  embeddings,
  embeddingType,
  onPointClick,
  selectedPoint,
  selectedArchetypes = [],
  colorBy = "archetype",
  colorScale,
  getAgeBin,
  width = 800,
  height = 600,
  margin = { top: 20, right: 20, bottom: 40, left: 40 },
}) => {
  const svgRef = useRef();
  const [tooltip, setTooltip] = useState(null);

  useEffect(() => {
    if (!data || !embeddings || !embeddingType) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove(); // Clear previous render

    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;

    // Create main group
    const g = svg
      .attr("width", width)
      .attr("height", height)
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    // Get current embedding data
    const currentEmbeddings = embeddings[embeddingType];

    // Create scales
    const xExtent = d3.extent(currentEmbeddings, (d) => d.x);
    const yExtent = d3.extent(currentEmbeddings, (d) => d.y);

    const xScale = d3
      .scaleLinear()
      .domain(xExtent)
      .range([0, plotWidth])
      .nice();

    const yScale = d3
      .scaleLinear()
      .domain(yExtent)
      .range([plotHeight, 0])
      .nice();

    // Use the provided color scale or create a default one
    const defaultColorScale = (value) => "#ccc";
    const actualColorScale = colorScale || defaultColorScale;

    // Add axes (no tick labels/markers)
    g.append("g")
      .attr("transform", `translate(0,${plotHeight})`)
      .call(d3.axisBottom(xScale).tickSize(0).tickFormat(""));

    g.append("g").call(d3.axisLeft(yScale).tickSize(0).tickFormat(""));

    // Add top and right axes
    g.append("g")
      .attr("transform", `translate(0,0)`)
      .call(d3.axisTop(xScale).tickSize(0).tickFormat(""));

    g.append("g")
      .attr("transform", `translate(${plotWidth},0)`)
      .call(d3.axisRight(yScale).tickSize(0).tickFormat(""));

    // Create tooltip
    const tooltipDiv = d3
      .select("body")
      .append("div")
      .attr("class", "tooltip")
      .style("opacity", 0)
      .style("position", "absolute")
      .style("background", "rgba(0, 0, 0, 0.8)")
      .style("color", "white")
      .style("padding", "8px")
      .style("border-radius", "4px")
      .style("font-size", "12px")
      .style("pointer-events", "none")
      .style("z-index", "1000");

    // Draw points
    const points = g
      .selectAll(".point")
      .data(currentEmbeddings)
      .enter()
      .append("circle")
      .attr("class", "point")
      .attr("cx", (d) => xScale(d.x))
      .attr("cy", (d) => yScale(d.y))
      .attr("r", (d, i) => {
        const isSelected = selectedPoint !== null && selectedPoint.index === i;
        return isSelected ? 8 : 4;
      })
      .attr("fill", (d, i) => {
        const persona = data[i];
        if (!persona) return "#ccc";

        let value;
        if (colorBy === "age" && getAgeBin) {
          value = getAgeBin(persona[colorBy]);
        } else {
          value = persona[colorBy];
        }

        return actualColorScale(value);
      })
      .attr("stroke", (d, i) => {
        const isSelected = selectedPoint !== null && selectedPoint.index === i;
        return isSelected ? "#000" : "#000";
      })
      .attr("stroke-width", (d, i) => {
        const isSelected = selectedPoint !== null && selectedPoint.index === i;
        return isSelected ? 2 : 0.5;
      })
      .style("cursor", "pointer")
      .style("opacity", (d, i) => {
        const persona = data[i];
        if (!persona) return 0.3;

        // If no items selected, show all
        if (selectedArchetypes.length === 0) return 1;

        // Get the value for comparison (age bin if applicable)
        let value;
        if (colorBy === "age" && getAgeBin) {
          value = getAgeBin(persona[colorBy]);
        } else {
          value = persona[colorBy];
        }

        // Show only selected items based on current color mapping
        return selectedArchetypes.includes(value) ? 1 : 0.2;
      })
      .on("mouseover", function (event, d) {
        const index = currentEmbeddings.indexOf(d);
        const persona = data[index];

        if (persona) {
          d3.select(this).attr("r", 8).style("opacity", 0.8);

          // Get the display value for the current color mapping
          let displayValue;
          if (colorBy === "age" && getAgeBin) {
            displayValue = getAgeBin(persona[colorBy]);
          } else {
            displayValue = persona[colorBy];
          }

          tooltipDiv.style("opacity", 1).html(`
              <strong>${persona.name}</strong><br/>
              ${persona.archetype}<br/>
              Age: ${persona.age}<br/>
              Location: ${persona.location}<br/>
              ${colorBy
                .replace("_", " ")
                .replace(/\b\w/g, (l) => l.toUpperCase())}: ${displayValue}
            `);
        }
      })
      .on("mousemove", function (event) {
        tooltipDiv
          .style("left", event.pageX + 10 + "px")
          .style("top", event.pageY - 10 + "px");
      })
      .on("mouseout", function () {
        d3.select(this).attr("r", 4).style("opacity", 1);

        tooltipDiv.style("opacity", 0);
      })
      .on("click", function (event, d) {
        const index = currentEmbeddings.indexOf(d);
        const persona = data[index];

        if (persona) {
          onPointClick({ ...persona, index });
        }
      });

    // Legend is now handled by separate InteractiveLegend component

    // Cleanup function
    return () => {
      tooltipDiv.remove();
    };
  }, [
    data,
    embeddings,
    embeddingType,
    selectedPoint,
    selectedArchetypes,
    colorBy,
    colorScale,
    width,
    height,
    margin,
  ]);

  return (
    <div className="scatter-plot-container w-full h-full flex items-center justify-center">
      <div className="max-w-full max-h-full overflow-auto">
        <svg ref={svgRef} className="max-w-full max-h-full" />
      </div>
    </div>
  );
};

export default ScatterPlot;
