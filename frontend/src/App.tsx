import { useState } from 'react';
import FileUpload from './components/FileUpload';
import ChatInterface from './components/ChatInterface';
import Histogram from './components/Histogram';
import CorrelationHeatmap from './components/CorrelationHeatmap';
import type { UploadResponse } from './types';
import { BarChart2, Database } from 'lucide-react';
import { MOCK_DATA } from './mockData';

function App() {
  const [data, setData] = useState<UploadResponse | null>(null);
  const [activeTab, setActiveTab] = useState<'distribution' | 'correlation'>('distribution');

  const handleUploadSuccess = (response: UploadResponse) => {
    setData(response);
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center space-x-3">
            <div className="bg-blue-600 p-2 rounded-lg">
              <BarChart2 className="w-6 h-6 text-white" />
            </div>
            <h1 className="text-xl font-bold text-gray-900">
              AI Data Analysis Platform
            </h1>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {!data ? (
          <div className="flex flex-col items-center justify-center space-y-8 py-12">
            <div className="text-center space-y-4 max-w-2xl">
              <h2 className="text-3xl font-bold text-gray-900">
                Upload your data to get started
              </h2>
              <p className="text-lg text-gray-600">
                Upload CSV, Excel, or JSON files. We'll automatically clean your data,
                detect types, and generate initial insights.
              </p>
            </div>
            <FileUpload onUploadSuccess={handleUploadSuccess} />

            <div className="relative">
              <div className="absolute inset-0 flex items-center">
                <div className="w-full border-t border-gray-300" />
              </div>
              <div className="relative flex justify-center text-sm">
                <span className="px-2 bg-gray-50 text-gray-500">Or for testing</span>
              </div>
            </div>

            <button
              onClick={() => setData(MOCK_DATA)}
              className="px-4 py-2 bg-white border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
            >
              Load Mock Dataset
            </button>
          </div>
        ) : (
          <div className="space-y-8">
            {/* Success Banner */}
            <div className="bg-green-50 border border-green-200 rounded-lg p-4 flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <div className="bg-green-100 p-2 rounded-full">
                  <Database className="w-5 h-5 text-green-600" />
                </div>
                <div>
                  <h3 className="font-medium text-green-900">Analysis Ready</h3>
                  <p className="text-sm text-green-700">
                    Successfully processed <strong>{data.filename}</strong>
                  </p>
                </div>
              </div>
              <button
                onClick={() => setData(null)}
                className="text-sm font-medium text-green-700 hover:text-green-800"
              >
                Upload New File
              </button>
            </div>

            {/* Metadata Grid */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
                <h3 className="text-sm font-medium text-gray-500 uppercase tracking-wider mb-2">Rows</h3>
                <p className="text-3xl font-bold text-gray-900">{data.metadata.rows.toLocaleString()}</p>
              </div>
              <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
                <h3 className="text-sm font-medium text-gray-500 uppercase tracking-wider mb-2">Columns</h3>
                <p className="text-3xl font-bold text-gray-900">{data.metadata.columns.toLocaleString()}</p>
              </div>
              <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
                <h3 className="text-sm font-medium text-gray-500 uppercase tracking-wider mb-2">Missing Values</h3>
                <p className="text-3xl font-bold text-gray-900">
                  {Object.values(data.metadata.missing_values).reduce((a, b) => a + b, 0)}
                </p>
              </div>
            </div>

            {/* Data Preview */}
            <div className="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden">
              <div className="px-6 py-4 border-b border-gray-100">
                <h3 className="font-semibold text-gray-900">Data Preview</h3>
              </div>
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      {data.metadata.column_names.map((col) => (
                        <th key={col} className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          {col}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {data.metadata.preview.map((row, idx) => (
                      <tr key={idx}>
                        {data.metadata.column_names.map((col) => (
                          <td key={`${idx}-${col}`} className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                            {String(row[col])}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Debug Info (Temporary) */}
            <div className="bg-gray-900 rounded-xl p-6 overflow-hidden">
              <h3 className="text-white font-medium mb-4">Raw Analysis Output</h3>
              <pre className="text-xs text-green-400 overflow-x-auto">
                {JSON.stringify(data.analysis, null, 2)}
              </pre>
            </div>

            {/* Chat Interface */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Left Column: Chat */}
              <div className="lg:col-span-2">
                <ChatInterface datasetId={data.dataset_id} filename={data.filename} />
              </div>

              {/* Right Column: Analysis Tabs */}
              <div className="lg:col-span-1 space-y-6">
                <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6 h-full overflow-y-auto max-h-[600px]">

                  {/* Tabs */}
                  <div className="flex space-x-1 bg-gray-100 p-1 rounded-lg mb-4">
                    <button
                      onClick={() => setActiveTab('distribution')}
                      className={`flex-1 py-1.5 text-sm font-medium rounded-md transition-colors ${activeTab === 'distribution'
                        ? 'bg-white text-gray-900 shadow-sm'
                        : 'text-gray-500 hover:text-gray-700'
                        }`}
                    >
                      Distribution
                    </button>
                    <button
                      onClick={() => setActiveTab('correlation')}
                      className={`flex-1 py-1.5 text-sm font-medium rounded-md transition-colors ${activeTab === 'correlation'
                        ? 'bg-white text-gray-900 shadow-sm'
                        : 'text-gray-500 hover:text-gray-700'
                        }`}
                    >
                      Correlation
                    </button>
                  </div>

                  {/* Distribution Content */}
                  {activeTab === 'distribution' && (
                    <div className="space-y-4">
                      <h3 className="font-semibold text-gray-900 mb-2">Distribution Analysis</h3>
                      {Object.entries(data.analysis.distribution_analysis || {}).map(([col, stats]) => (
                        <div key={col} className="border border-gray-200 rounded-lg p-4">
                          <h4 className="font-medium text-gray-800 mb-2">{col}</h4>
                          <div className="space-y-1 text-sm text-gray-600">
                            <div className="flex justify-between">
                              <span>Skewness:</span>
                              <span className="font-mono">{stats.skewness?.toFixed(2) ?? 'N/A'}</span>
                            </div>
                            <div className="flex justify-between">
                              <span>Kurtosis:</span>
                              <span className="font-mono">{stats.kurtosis?.toFixed(2) ?? 'N/A'}</span>
                            </div>
                            <div className="flex justify-between">
                              <span>Outliers:</span>
                              <span className="font-mono text-red-500">{stats.outlier_count} ({stats.outlier_percentage.toFixed(1)}%)</span>
                            </div>
                          </div>
                          <div className="mt-3 pt-3 border-t border-gray-100">
                            <p className="text-xs text-gray-400 mb-1">Distribution</p>
                            <Histogram data={stats.histogram} />
                          </div>
                        </div>
                      ))}
                    </div>
                  )}

                  {/* Correlation Content */}
                  {activeTab === 'correlation' && (
                    <div>
                      <h3 className="font-semibold text-gray-900 mb-4">Correlation Matrix</h3>
                      <CorrelationHeatmap matrix={data.analysis.correlation_matrix} />
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
