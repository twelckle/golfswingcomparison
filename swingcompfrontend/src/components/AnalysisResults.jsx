// src/components/AnalysisResults.jsx
import React from "react";

const AnalysisResults = ({ results, sliderIndex, setSliderIndex }) => {
    if (!results) return null;

    return (
        <div className="mt-6 text-center w-full max-w-screen-xl px-4 mx-auto">
            <h2 className="text-xl font-semibold text-black">
                From our database your swing is most similar to...
            </h2>
            <p className="text-4xl font-bold text-gray-800">
                {results.match} ({results.similarity}% similarity)
            </p>

            {results.frames?.length > 0 && results.pro_frames?.length > 0 && (
                <div className="mt-6 w-full">
                    <div className="flex flex-col md:flex-row justify-center items-center gap-6">
                        <div className="flex-1 text-center">
                            <p className="text-xl font-semibold text-gray-700 mb-1">Your Swing</p>
                            <img
                                src={results.frames[sliderIndex]}
                                alt={`User Frame`}
                                className="w-full h-auto max-h-[70vh] object-contain rounded shadow"
                            />
                        </div>
                        <div className="flex-1 text-center">
                            <p className="text-xl font-semibold text-gray-700 mb-1">{results.match}</p>
                            <img
                                src={results.pro_frames[sliderIndex]}
                                alt={`Pro Frame`}
                                className="w-full h-auto max-h-[70vh] object-contain rounded shadow"
                            />
                        </div>
                    </div>
                    <div className="mt-4 text-center">
                        <input
                            type="range"
                            min={0}
                            max={results.frames.length - 1}
                            value={sliderIndex}
                            onChange={(e) => setSliderIndex(Number(e.target.value))}
                            className="w-full md:w-2/3"
                        />
                        <p className="text-sm text-gray-500 mt-1">
                            Frame {sliderIndex + 1} / {results.frames.length}
                        </p>
                    </div>
                </div>
            )}

            <div className="mt-10 text-center">
                <h3 className="text-lg font-semibold text-gray-700 mb-2">Overlay View</h3>
                <div className="relative w-full max-w-xl mx-auto aspect-[4/3]">
                    <img
                        src={results.overlay_frames[sliderIndex]}
                        alt={`Overlay Frame`}
                        className="absolute top-0 left-0 w-full h-full object-contain"
                    />
                </div>
                <p className="text-sm text-gray-500 mt-1">
                    Frame {sliderIndex + 1} / {results.frames?.length || 0} (Overlay)
                </p>
            </div>
        </div>
    );
};

export default AnalysisResults;