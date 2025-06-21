import React from "react";
import "./LandingPage.css"; // Optional, if you want custom styles

function LandingPage({ onTryNowClick }) {
    return (
        <div
            className="landing-container"
            style={{ paddingTop: "4rem", minHeight: "calc(100vh - 224px)", paddingBottom: "0", boxSizing: "border-box" }}
        >

            <main className="landing-main" style={{
                display: "flex",
                flexDirection: "row",
                justifyContent: "center",
                alignItems: "stretch",
                gap: "10rem",
                flexWrap: "wrap",
                width: "100%",
                maxWidth: "1400px",
                margin: "0 auto"
            }}>
                <div className="landing-content" style={{
                    display: "flex",
                    flex: 1,
                    flexDirection: "row",
                    gap: "2rem",
                    justifyContent: "center",
                    alignItems: "center",
                    width: "100%",
                    flexWrap: "wrap"
                }}>
                    <div style={{
                        flex: "1",
                        minWidth: "48%",
                        display: "flex",
                        flexDirection: "column",
                        justifyContent: "space-between",
                        alignItems: "center",
                        textAlign: "center",
                        padding: "1rem"
                    }}>
                        <div style={{
                            display: "flex",
                            flexDirection: "column",
                            justifyContent: "center",
                            height: "100%"
                        }}>
                            <div style={{ fontSize: "1.2rem", maxWidth: "600px" }}>
                                <div className="landing-text" style={{ fontSize: "1.5rem" }}>
                                    <p>
                                        Upload a short video, and our tool will break down your swing using AI-powered pose detection.
                                        You'll see a side-by-side, frame-by-frame comparison with a professional golfer, along with visual feedback
                                        and similarity scores. It’s a fun way to notice how your swing compares — and maybe even spot something new in your form.
                                    </p>
                                </div>
                            </div>
                            <div style={{ textAlign: "center", marginTop: "1.5rem" }}>
                                <p style={{ marginBottom: "0.5rem", fontSize: "1.3rem" }}>
                                    Upload a ~30sec <strong>slow motion</strong> video of your iron swing <br /> in portrait mode for best results
                                </p>
                            </div>
                        </div>
                        <div className="try-now-section" style={{ display: "flex", justifyContent: "center", paddingBottom: "0" }}>
                            <button onClick={onTryNowClick} className="try-now-button">
                                Try Now
                            </button>
                        </div>
                    </div>
                    <div className="landing-video" style={{
                        flex: "1",
                        minWidth: "48%",
                        display: "flex",
                        justifyContent: "center",
                        alignItems: "center",
                        alignSelf: "center",
                        marginTop: "0"
                    }}>
                        <iframe
                            width="100%"
                            height="460"
                            src="https://www.youtube.com/embed/gVa3cnb_DG8?si=o2kQG68zh01OB7rP"
                            title="Swing Comparison Demo"
                            frameBorder="0"
                            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                            allowFullScreen
                        ></iframe>
                    </div>
                </div>
            </main>
        </div>
    );
}

export default LandingPage;