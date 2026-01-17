import type { UploadResponse } from './types';

export const MOCK_DATA: UploadResponse = {
    dataset_id: "mock-dataset-123",
    filename: "sales_data_2024.csv",
    message: "Successfully loaded mock data",
    metadata: {
        rows: 1000,
        columns: 5,
        column_names: ["Date", "Product", "Region", "Sales", "Profit"],
        dtypes: {
            "Date": "datetime64[ns]",
            "Product": "object",
            "Region": "object",
            "Sales": "float64",
            "Profit": "float64"
        },
        missing_values: {
            "Date": 0,
            "Product": 0,
            "Region": 0,
            "Sales": 0,
            "Profit": 0
        },
        preview: [
            { "Date": "2024-01-01", "Product": "Laptop", "Region": "North", "Sales": 1200.50, "Profit": 300.20 },
            { "Date": "2024-01-02", "Product": "Mouse", "Region": "South", "Sales": 25.00, "Profit": 5.00 },
            { "Date": "2024-01-03", "Product": "Monitor", "Region": "East", "Sales": 350.00, "Profit": 80.50 },
            { "Date": "2024-01-04", "Product": "Keyboard", "Region": "West", "Sales": 80.00, "Profit": 20.00 },
            { "Date": "2024-01-05", "Product": "Laptop", "Region": "North", "Sales": 1150.00, "Profit": 280.00 }
        ]
    },
    analysis: {
        descriptive_statistics: {
            "Sales": { "mean": 500, "std": 200, "min": 20, "max": 2000 },
            "Profit": { "mean": 100, "std": 50, "min": 5, "max": 500 }
        },
        correlation_matrix: {
            "Sales": { "Sales": 1.0, "Profit": 0.85 },
            "Profit": { "Sales": 0.85, "Profit": 1.0 }
        },
        distribution_analysis: {
            "Sales": {
                skewness: 0.5,
                kurtosis: -0.2,
                outlier_count: 5,
                outlier_percentage: 0.5,
                histogram: [
                    { bin_start: 0, bin_end: 200, count: 150, label: "0-200" },
                    { bin_start: 200, bin_end: 400, count: 300, label: "200-400" },
                    { bin_start: 400, bin_end: 600, count: 250, label: "400-600" },
                    { bin_start: 600, bin_end: 800, count: 200, label: "600-800" },
                    { bin_start: 800, bin_end: 1000, count: 100, label: "800-1000" }
                ]
            },
            "Profit": {
                skewness: 0.2,
                kurtosis: 0.1,
                outlier_count: 2,
                outlier_percentage: 0.2,
                histogram: [
                    { bin_start: 0, bin_end: 50, count: 200, label: "0-50" },
                    { bin_start: 50, bin_end: 100, count: 400, label: "50-100" },
                    { bin_start: 100, bin_end: 150, count: 300, label: "100-150" },
                    { bin_start: 150, bin_end: 200, count: 100, label: "150-200" }
                ]
            }
        }
    }
};
