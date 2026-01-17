import React from 'react';
import { BarChart, Bar, Tooltip, ResponsiveContainer } from 'recharts';

interface HistogramProps {
    data: Array<{
        bin_start: number;
        bin_end: number;
        count: number;
        label: string;
    }>;
    color?: string;
}

const Histogram: React.FC<HistogramProps> = ({ data, color = "#3b82f6" }) => {
    return (
        <div className="h-24 w-full">
            <ResponsiveContainer width="100%" height="100%">
                <BarChart data={data}>
                    <Tooltip
                        contentStyle={{ fontSize: '12px', padding: '4px' }}
                        cursor={{ fill: '#f3f4f6' }}
                    />
                    <Bar dataKey="count" fill={color} radius={[2, 2, 0, 0]} />
                </BarChart>
            </ResponsiveContainer>
        </div>
    );
};

export default Histogram;
