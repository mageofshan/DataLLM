export interface DatasetMetadata {
    rows: number;
    columns: number;
    column_names: string[];
    dtypes: Record<string, string>;
    missing_values: Record<string, number>;
    preview: Record<string, any>[];
}

export interface BasicStats {
    descriptive_statistics: Record<string, any>;
    correlation_matrix: Record<string, Record<string, number>>;
    distribution_analysis: Record<string, {
        skewness: number | null;
        kurtosis: number | null;
        outlier_count: number;
        outlier_percentage: number;
        histogram: Array<{
            bin_start: number;
            bin_end: number;
            count: number;
            label: string;
        }>;
    }>;
}

export interface UploadResponse {
    dataset_id: string;
    filename: string;
    metadata: DatasetMetadata;
    analysis: BasicStats;
    message: string;
}
