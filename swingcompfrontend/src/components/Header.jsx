// src/components/Header.jsx
import React from 'react';

function Header() {
    return (
        <>
            {/* Logo fixed in top-left */}
            <div className="absolute top-0 left-0 p-4">
                <img src="/logo.png" alt="Golf Swing Logo" className="h-65 w-65" />
            </div>

            {/* Download button in top-right */}
            <div className="absolute top-0 right-0 p-4">
                <a
                    href="/MySwing.mp4"
                    download="TheoSwing_example.mp4"
                    className="try-now-button flex items-center gap-2"
                    style={{ padding: '0.5rem 1.5rem', fontSize: '1rem' }}
                >
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2M7 10l5 5m0 0l5-5m-5 5V4" />
                    </svg>
                    Download Example Swing
                </a>
            </div>

            {/* Centered headline */}
            <div className="flex justify-center mt-24 mb-16">
                <h1
                    style={{
                        fontFamily: "'Bebas Neue', 'arial black', sans-serif",
                        fontSize: "4rem",
                        fontWeight: "bold",
                        letterSpacing: "0.05em",
                        textShadow: "2px 2px 4px rgba(0, 0, 0, 0.2)",
                        textTransform: "uppercase"
                    }}
                >
                    Swing like a pro?
                </h1>
            </div>
        </>
    );
}

export default Header;
