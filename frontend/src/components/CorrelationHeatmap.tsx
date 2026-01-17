import React from 'react';

interface CorrelationHeatmapProps {
    matrix: Record<string, Record<string, number>>;
}

const CorrelationHeatmap: React.FC<CorrelationHeatmapProps> = ({ matrix }) => {
    const variables = Object.keys(matrix);

    if (variables.length === 0) {
        return <div className="text-gray-500 text-sm text-center p-4">No numeric correlation data available.</div>;
    }

    // Helper to get color for correlation value
    const getColor = (value: number) => {
        // -1 -> Red (255, 0, 0)
        // 0 -> White (255, 255, 255)
        // 1 -> Blue (0, 0, 255)

        if (value > 0) {
            // White to Blue
            const intensity = Math.round(255 * (1 - value));
            return `rgb(${intensity}, ${intensity}, 255)`;
        } else {
            // White to Red
            const intensity = Math.round(255 * (1 + value)); // value is negative
            return `rgb(255, ${intensity}, ${intensity})`;
        }
    };

    return (
        <div className="overflow-x-auto">
            <div className="inline-block min-w-full">
                <div className="grid" style={{
                    gridTemplateColumns: `auto repeat(${variables.length}, minmax(40px, 1fr))`
                }}>
                    {/* Header Row */}
                    <div className="h-10"></div> {/* Top-left corner */}
                    {variables.map((v) => (
                        <div key={v} className="h-10 flex items-center justify-center p-1">
                            <span className="text-xs font-medium text-gray-600 truncate w-full text-center" title={v}>
                                {v}
                            </span>
                        </div>
                    ))}

                    {/* Data Rows */}
                    {variables.map((rowVar) => (
                        <React.Fragment key={rowVar}>
                            {/* Row Label */}
                            <div className="h-10 flex items-center justify-end p-2">
                                <span className="text-xs font-medium text-gray-600 truncate max-w-[100px]" title={rowVar}>
                                    {rowVar}
                                </span>
                            </div>

                            {/* Cells */}
                            {variables.map((colVar) => {
                                const value = matrix[rowVar][colVar];
                                return (
                                    <div
                                        key={`${rowVar}-${colVar}`}
                                        className="h-10 border border-gray-100 flex items-center justify-center relative group"
                                        style={{ backgroundColor: getColor(value) }}
                                    >
                                        <span className="text-[10px] text-gray-700 font-mono opacity-0 group-hover:opacity-100 transition-opacity">
                                            {value.toFixed(2)}
                                        </span>
                                        <div className="absolute inset-0 opacity-0 group-hover:opacity-100 bg-black/5 pointer-events-none" />
                                        <div className="absolute bottom-full mb-1 hidden group-hover:block bg-gray-800 text-white text-xs p-1 rounded z-10 whitespace-nowrap">
                                            {rowVar} vs {colVar}: {value.toFixed(3)}
                                        </div>
                                    </div>
                                );
                            })}
                        </React.Fragment>
                    ))}
                </div>
            </div>
        </div>
    );
};

export default CorrelationHeatmap;
