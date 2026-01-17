import React from 'react';
import {
    BarChart,
    Bar,
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    Legend,
    ResponsiveContainer,
    ScatterChart,
    Scatter,
    ComposedChart
} from 'recharts';

interface ChartData {
    type: 'bar' | 'line' | 'scatter' | 'linear_regression';
    title: string;
    data: any[];
    xKey: string;
    yKey: string;
}

interface ChartRendererProps {
    config: ChartData;
}

const ChartRenderer: React.FC<ChartRendererProps> = ({ config }) => {
    const renderChart = () => {
        switch (config.type) {
            case 'bar':
                return (
                    <BarChart data={config.data}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} />
                        <XAxis dataKey={config.xKey} tick={{ fontSize: 12 }} interval={0} />
                        <YAxis tick={{ fontSize: 12 }} />
                        <Tooltip
                            contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
                            cursor={{ fill: '#f3f4f6' }}
                        />
                        <Legend />
                        <Bar dataKey={config.yKey} fill="#3b82f6" radius={[4, 4, 0, 0]} name={config.yKey} />
                    </BarChart>
                );
            case 'line':
                return (
                    <LineChart data={config.data}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} />
                        <XAxis dataKey={config.xKey} tick={{ fontSize: 12 }} />
                        <YAxis tick={{ fontSize: 12 }} />
                        <Tooltip
                            contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
                        />
                        <Legend />
                        <Line type="monotone" dataKey={config.yKey} stroke="#3b82f6" strokeWidth={2} dot={{ r: 4 }} activeDot={{ r: 6 }} name={config.yKey} />
                    </LineChart>
                );
            case 'scatter':
                return (
                    <ScatterChart>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis type="category" dataKey={config.xKey} name={config.xKey} tick={{ fontSize: 12 }} />
                        <YAxis type="number" dataKey={config.yKey} name={config.yKey} tick={{ fontSize: 12 }} />
                        <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                        <Legend />
                        <Scatter name={config.title} data={config.data} fill="#3b82f6" />
                    </ScatterChart>
                );
            case 'linear_regression':
                return (
                    <ComposedChart data={config.data}>
                        <CartesianGrid strokeDasharray="3 3" />
                        {/* XAxis type depends on data. For regression usually number. */}
                        <XAxis dataKey={config.xKey} type="category" tick={{ fontSize: 12 }} allowDuplicatedCategory={false} />
                        <YAxis tick={{ fontSize: 12 }} />
                        <Tooltip />
                        <Legend />
                        <Scatter name="Data Points" dataKey={config.yKey} fill="#3b82f6" />
                        {/* Assuming data has y_pred for the line */}
                        <Line type="monotone" dataKey="y_pred" stroke="#ff7300" dot={false} strokeWidth={2} name="Regression Line" />
                    </ComposedChart>
                );
            default:
                return <div className="flex items-center justify-center h-full text-gray-500">Unsupported chart type</div>;
        }
    };

    return (
        <div className="w-full h-full flex flex-col">
            <h4 className="text-sm font-semibold text-gray-700 text-center mb-4">{config.title}</h4>
            <div className="flex-1 min-h-0">
                <ResponsiveContainer width="100%" height="100%" minHeight={200}>
                    {renderChart()}
                </ResponsiveContainer>
            </div>
        </div>
    );
};

export default ChartRenderer;
