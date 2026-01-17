import React, { useState } from 'react';
import {
    X,
    Table as TableIcon,
    Activity
} from 'lucide-react';
import {
    ComposedChart,
    Line,
    Scatter,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    Legend,
    ResponsiveContainer
} from 'recharts';

interface VisualizationModalProps {
    isOpen: boolean;
    onClose: () => void;
    data: {
        type: string;
        title: string;
        x_label: string;
        y_label: string;
        data: any[];
        metadata?: any;
    };
}

const VisualizationModal: React.FC<VisualizationModalProps> = ({ isOpen, onClose, data }) => {
    const [activeView, setActiveView] = useState<'chart' | 'table'>('chart');

    if (!isOpen) return null;

    const renderChart = () => {
        if (data.type === 'linear_regression' || data.type === 'scatter') {
            // Sort data for line rendering if needed
            const sortedData = [...data.data].sort((a, b) => {
                if (typeof a.x === 'number' && typeof b.x === 'number') return a.x - b.x;
                return 0;
            });

            return (
                <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={sortedData} margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis
                            dataKey="x"
                            type="number"
                            label={{ value: data.x_label, position: 'insideBottomRight', offset: -10 }}
                            domain={['auto', 'auto']}
                        />
                        <YAxis
                            label={{ value: data.y_label, angle: -90, position: 'insideLeft' }}
                        />
                        <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                        <Legend />
                        <Scatter name="Observed" dataKey="y" fill="#3b82f6" />
                        {data.type === 'linear_regression' && (
                            <Line
                                type="monotone"
                                dataKey="y_pred"
                                stroke="#ef4444"
                                dot={false}
                                strokeWidth={2}
                                name="Prediction"
                                activeDot={false}
                            />
                        )}
                    </ComposedChart>
                </ResponsiveContainer>
            );
        }
        return <div className="flex items-center justify-center h-full">Unsupported chart type: {data.type}</div>;
    };

    return (
        <div className="fixed inset-0 z-50 overflow-y-auto">
            <div className="flex items-center justify-center min-h-screen px-4 pt-4 pb-20 text-center sm:block sm:p-0">
                <div className="fixed inset-0 transition-opacity" aria-hidden="true">
                    <div className="absolute inset-0 bg-gray-500 opacity-75" onClick={onClose}></div>
                </div>

                <span className="hidden sm:inline-block sm:align-middle sm:h-screen" aria-hidden="true">&#8203;</span>

                <div className="inline-block align-bottom bg-white rounded-lg text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-4xl sm:w-full">
                    <div className="bg-white px-4 pt-5 pb-4 sm:p-6 sm:pb-4">
                        <div className="flex justify-between items-start mb-6">
                            <div>
                                <h3 className="text-lg leading-6 font-medium text-gray-900" id="modal-title">
                                    {data.title}
                                </h3>
                                {data.metadata?.equation && (
                                    <p className="text-sm text-gray-500 mt-1">
                                        Equation: <span className="font-mono bg-gray-100 px-1 rounded">{data.metadata.equation}</span>
                                        {data.metadata.r2 && <span> • R²: {data.metadata.r2.toFixed(4)}</span>}
                                    </p>
                                )}
                            </div>
                            <div className="flex items-center space-x-2">
                                <div className="flex bg-gray-100 p-1 rounded-lg">
                                    <button
                                        onClick={() => setActiveView('chart')}
                                        className={`p-1.5 rounded-md transition-all ${activeView === 'chart' ? 'bg-white shadow text-blue-600' : 'text-gray-500 hover:text-gray-700'}`}
                                        title="Chart View"
                                    >
                                        <Activity className="w-4 h-4" />
                                    </button>
                                    <button
                                        onClick={() => setActiveView('table')}
                                        className={`p-1.5 rounded-md transition-all ${activeView === 'table' ? 'bg-white shadow text-blue-600' : 'text-gray-500 hover:text-gray-700'}`}
                                        title="Table View"
                                    >
                                        <TableIcon className="w-4 h-4" />
                                    </button>
                                </div>
                                <button onClick={onClose} className="text-gray-400 hover:text-gray-500">
                                    <X className="w-6 h-6" />
                                </button>
                            </div>
                        </div>

                        <div className="h-[500px] w-full bg-gray-50 rounded-xl border border-gray-200 p-4">
                            {activeView === 'chart' ? (
                                renderChart()
                            ) : (
                                <div className="h-full overflow-auto">
                                    <table className="min-w-full divide-y divide-gray-200">
                                        <thead className="bg-gray-50 sticky top-0">
                                            <tr>
                                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">{data.x_label} (X)</th>
                                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">{data.y_label} (Y)</th>
                                                {data.type === 'linear_regression' && (
                                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Predicted (Y')</th>
                                                )}
                                                {data.type === 'linear_regression' && (
                                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Residual</th>
                                                )}
                                            </tr>
                                        </thead>
                                        <tbody className="bg-white divide-y divide-gray-200">
                                            {data.data.map((row, idx) => (
                                                <tr key={idx}>
                                                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{typeof row.x === 'number' ? row.x.toFixed(2) : row.x}</td>
                                                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{typeof row.y === 'number' ? row.y.toFixed(2) : row.y}</td>
                                                    {data.type === 'linear_regression' && (
                                                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{row.y_pred?.toFixed(2)}</td>
                                                    )}
                                                    {data.type === 'linear_regression' && (
                                                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{(row.y - row.y_pred)?.toFixed(4)}</td>
                                                    )}
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default VisualizationModal;
