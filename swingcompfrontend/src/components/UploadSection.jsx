import React, { useState, useEffect } from 'react';
import axios from 'axios';

const UploadSection = ({ onAnalysisComplete }) => {
    const [file, setFile] = useState(null);
    const [preview, setPreview] = useState(null);
    const [loading, setLoading] = useState(false);
    const [ellipsis, setEllipsis] = useState('.');

    const handleDrop = (e) => {
        e.preventDefault();
        const droppedFile = e.dataTransfer.files[0];
        setFile(droppedFile);
        setPreview(URL.createObjectURL(droppedFile));
    };

    const handleDragOver = (e) => {
        e.preventDefault();
    };

    const handleUpload = async () => {
        if (!file) return;
        setLoading(true);

        try {
            const formData = new FormData();
            formData.append('video', file);

            const response = await axios.post('http://127.0.0.1:5000/analyze', formData, {
                headers: { 'Content-Type': 'multipart/form-data' },
            });

            if (onAnalysisComplete) {
                onAnalysisComplete(response.data);
            }
        } catch (error) {
            console.error('Upload failed:', error);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        if (!loading) return;

        const interval = setInterval(() => {
            setEllipsis((prev) => (prev.length >= 3 ? '.' : prev + '.'));
        }, 500);

        return () => clearInterval(interval);
    }, [loading]);

    return (
        <section id="upload" className="flex flex-col items-center py-20 bg-gray-100">
            <h2 className="text-3xl font-semibold mb-6">Upload Your Swing</h2>
            {!preview && (
                <div
                    className="w-[30rem] h-56 border-4 border-dashed border-gray-400 flex items-center justify-center mb-4 rounded"
                    onDrop={handleDrop}
                    onDragOver={handleDragOver}
                >
                    <p>Drag & drop a swing video here</p>
                </div>
            )}
            {preview && (
                <div className="mb-4">
                    <video src={preview} controls className="w-96 rounded" />
                </div>
            )}
            <button
                onClick={handleUpload}
                className="px-6 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition"
            >
                <span className="inline-block w-32 text-center">
                    {loading ? `Analyzing${ellipsis}` : 'Upload'}
                </span>
            </button>
        </section>
    );
};

export default UploadSection;