import axios from 'axios';
import type { UploadResponse } from '../types';

const API_URL = 'http://localhost:8000/api/v1';

const api = axios.create({
    baseURL: API_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

export const uploadFile = async (file: File): Promise<UploadResponse> => {
    const formData = new FormData();
    formData.append('file', file);

    const response = await api.post<UploadResponse>('/upload', formData, {
        headers: {
            'Content-Type': 'multipart/form-data',
        },
    });
    return response.data;
};

export interface ChatRequest {
    query: string;
    dataset_id?: string;
    session_id?: string;
}

export interface ChatResponse {
    response: string;
    route: string;
    data?: any;
    session_id: string;
}

export const sendChatMessage = async (payload: ChatRequest): Promise<ChatResponse> => {
    const response = await api.post<ChatResponse>('/chat', payload);
    return response.data;
};
