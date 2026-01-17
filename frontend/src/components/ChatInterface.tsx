import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, Loader2, BarChart2 } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { sendChatMessage } from '../services/api';
import ChartRenderer from './ChartRenderer';
import VisualizationModal from './VisualizationModal';

interface ChatInterfaceProps {
    datasetId: string;
    filename: string;
}

interface Message {
    id: string;
    role: 'user' | 'assistant';
    content: string;
    data?: any;
}

const ChatInterface: React.FC<ChatInterfaceProps> = ({ datasetId, filename }) => {
    const [messages, setMessages] = useState<Message[]>([
        {
            id: 'welcome',
            role: 'assistant',
            content: `Hello! I've analyzed **${filename}**. You can ask me questions about the data, like "What is the average of column X?" or "Show me the correlation between A and B".`
        }
    ]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [sessionId, setSessionId] = useState<string | undefined>(undefined);
    const [selectedViz, setSelectedViz] = useState<any>(null);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const handleSend = async () => {
        if (!input.trim() || isLoading) return;

        const userMessage: Message = {
            id: Date.now().toString(),
            role: 'user',
            content: input
        };

        setMessages(prev => [...prev, userMessage]);
        setInput('');
        setIsLoading(true);

        try {
            const response = await sendChatMessage({
                query: userMessage.content,
                dataset_id: datasetId,
                session_id: sessionId
            });

            setSessionId(response.session_id);

            const botMessage: Message = {
                id: (Date.now() + 1).toString(),
                role: 'assistant',
                content: response.response,
                data: response.data
            };

            setMessages(prev => [...prev, botMessage]);
        } catch (error) {
            console.error("Chat error:", error);
            const errorMessage: Message = {
                id: (Date.now() + 1).toString(),
                role: 'assistant',
                content: "Sorry, I encountered an error processing your request."
            };
            setMessages(prev => [...prev, errorMessage]);
        } finally {
            setIsLoading(false);
        }
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    return (
        <div className="flex flex-col h-[600px] bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
            {/* Chat Header */}
            <div className="p-4 border-b border-gray-200 bg-gray-50 flex items-center justify-between">
                <div className="flex items-center space-x-2">
                    <Bot className="w-5 h-5 text-blue-600" />
                    <span className="font-medium text-gray-700">Data Assistant</span>
                </div>
                <span className="text-xs text-gray-500 bg-gray-200 px-2 py-1 rounded-full">
                    {filename}
                </span>
            </div>

            {/* Messages Area */}
            <div className="flex-1 overflow-y-auto p-4 space-y-6">
                {messages.map((msg) => (
                    <div
                        key={msg.id}
                        className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                    >
                        <div
                            className={`flex max-w-[80%] ${msg.role === 'user' ? 'flex-row-reverse' : 'flex-row'
                                } items-start gap-3`}
                        >
                            <div
                                className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${msg.role === 'user' ? 'bg-blue-600' : 'bg-green-600'
                                    }`}
                            >
                                {msg.role === 'user' ? (
                                    <User className="w-5 h-5 text-white" />
                                ) : (
                                    <Bot className="w-5 h-5 text-white" />
                                )}
                            </div>

                            <div
                                className={`p-4 rounded-2xl ${msg.role === 'user'
                                    ? 'bg-blue-600 text-white rounded-tr-none'
                                    : 'bg-gray-100 text-gray-800 rounded-tl-none'
                                    }`}
                            >
                                <div className="prose prose-sm max-w-none dark:prose-invert">
                                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                        {msg.content.replace(/```json[\s\S]*?```/g, '')}
                                    </ReactMarkdown>
                                </div>

                                {/* Render Chart if present in content (parsed from JSON block) */}
                                {(() => {
                                    // 1. Check for structured visualization data from backend
                                    if (msg.data?.visualization) {
                                        return (
                                            <div className="mt-4 bg-white p-4 rounded-lg border border-gray-200 h-auto w-full">
                                                <div className="h-80 w-full">
                                                    <ChartRenderer config={{
                                                        type: msg.data.visualization.type,
                                                        title: msg.data.visualization.title,
                                                        data: msg.data.visualization.data,
                                                        xKey: 'x',
                                                        yKey: 'y'
                                                    }} />
                                                </div>
                                                {msg.data.visualization.equation && (
                                                    <p className="text-xs text-center text-gray-500 mt-2 font-mono bg-gray-50 py-1 rounded">
                                                        Equation: {msg.data.visualization.equation}
                                                    </p>
                                                )}
                                            </div>
                                        );
                                    }

                                    // 2. Fallback: Check for JSON block in content (Legacy)
                                    const match = msg.content.match(/```json\n([\s\S]*?)\n```/);
                                    if (match) {
                                        try {
                                            const chartData = JSON.parse(match[1]);
                                            if (chartData.type && chartData.data) {
                                                return (
                                                    <div className="mt-4 bg-white p-4 rounded-lg border border-gray-200 h-80 w-full">
                                                        <ChartRenderer config={chartData} />
                                                    </div>
                                                );
                                            }
                                        } catch (e) {
                                            console.error("Failed to parse chart JSON", e);
                                        }
                                    }
                                    return null;
                                })()}

                                {/* Render Interactive Visualization Card */}
                                {msg.data?.visualization && (
                                    <div className="mt-4">
                                        <button
                                            onClick={() => setSelectedViz(msg.data.visualization)}
                                            className="flex items-center gap-3 p-4 bg-blue-50 border border-blue-100 rounded-xl hover:bg-blue-100 transition-colors w-full text-left group"
                                        >
                                            <div className="bg-blue-100 p-2 rounded-lg group-hover:bg-white transition-colors">
                                                <BarChart2 className="w-5 h-5 text-blue-600" />
                                            </div>
                                            <div>
                                                <h4 className="font-medium text-blue-900">{msg.data.visualization.title}</h4>
                                                <p className="text-xs text-blue-700 mt-1">Click to view interactive graph and data</p>
                                            </div>
                                        </button>
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                ))}
                {isLoading && (
                    <div className="flex justify-start">
                        <div className="flex items-center gap-3">
                            <div className="w-8 h-8 rounded-full bg-green-600 flex items-center justify-center">
                                <Loader2 className="w-5 h-5 text-white animate-spin" />
                            </div>
                            <div className="bg-gray-100 p-4 rounded-2xl rounded-tl-none text-gray-500 text-sm">
                                Thinking...
                            </div>
                        </div>
                    </div>
                )}
                <div ref={messagesEndRef} />
            </div>

            {/* Input Area */}
            <div className="p-4 border-t border-gray-200 bg-white">
                <div className="flex items-center gap-2">
                    <input
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={handleKeyDown}
                        placeholder="Ask a question about your data..."
                        className="flex-1 p-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                        disabled={isLoading}
                    />
                    <button
                        onClick={handleSend}
                        disabled={isLoading || !input.trim()}
                        className="p-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                    >
                        <Send className="w-5 h-5" />
                    </button>
                </div>
            </div>

            {/* Visualization Modal */}
            <VisualizationModal
                isOpen={!!selectedViz}
                onClose={() => setSelectedViz(null)}
                data={selectedViz}
            />
        </div>
    );
};

export default ChatInterface;
