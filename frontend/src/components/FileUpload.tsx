import React, { useState, useCallback } from 'react';
import { Upload, AlertCircle, Loader2 } from 'lucide-react';
import { uploadFile } from '../services/api';
import type { UploadResponse } from '../types';

interface FileUploadProps {
    onUploadSuccess: (data: UploadResponse) => void;
}

const FileUpload: React.FC<FileUploadProps> = ({ onUploadSuccess }) => {
    const [isDragging, setIsDragging] = useState(false);
    const [isUploading, setIsUploading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const handleDrag = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === 'dragenter' || e.type === 'dragover') {
            setIsDragging(true);
        } else if (e.type === 'dragleave') {
            setIsDragging(false);
        }
    }, []);

    const processFile = async (file: File) => {
        if (!file) return;

        // Simple validation
        // Note: MIME types can be tricky, checking extension is also good practice in production

        setIsUploading(true);
        setError(null);

        try {
            const data = await uploadFile(file);
            onUploadSuccess(data);
        } catch (err: any) {
            console.error("Upload failed", err);
            setError(err.response?.data?.detail || "Failed to upload file. Please try again.");
        } finally {
            setIsUploading(false);
        }
    };

    const handleDrop = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragging(false);

        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            processFile(e.dataTransfer.files[0]);
        }
    }, []);

    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            processFile(e.target.files[0]);
        }
    };

    return (
        <div className="w-full max-w-xl mx-auto">
            <div
                className={`relative border-2 border-dashed rounded-xl p-8 transition-all duration-200 ease-in-out text-center
          ${isDragging
                        ? 'border-blue-500 bg-blue-50 scale-[1.02]'
                        : 'border-gray-300 hover:border-blue-400 hover:bg-gray-50'
                    }
          ${isUploading ? 'opacity-50 pointer-events-none' : ''}
        `}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
            >
                <input
                    type="file"
                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                    onChange={handleChange}
                    accept=".csv,.xlsx,.json,.tsv"
                    disabled={isUploading}
                />

                <div className="flex flex-col items-center justify-center space-y-4">
                    <div className={`p-4 rounded-full ${isDragging ? 'bg-blue-100' : 'bg-gray-100'}`}>
                        {isUploading ? (
                            <Loader2 className="w-8 h-8 text-blue-600 animate-spin" />
                        ) : (
                            <Upload className={`w-8 h-8 ${isDragging ? 'text-blue-600' : 'text-gray-500'}`} />
                        )}
                    </div>

                    <div className="space-y-1">
                        <p className="text-lg font-medium text-gray-700">
                            {isUploading ? 'Processing your data...' : 'Drop your dataset here'}
                        </p>
                        <p className="text-sm text-gray-500">
                            or click to browse (CSV, Excel, JSON)
                        </p>
                    </div>
                </div>
            </div>

            {error && (
                <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg flex items-start space-x-3">
                    <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
                    <p className="text-sm text-red-700">{error}</p>
                </div>
            )}
        </div>
    );
};

export default FileUpload;
