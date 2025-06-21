import React, { useRef, useState, useEffect } from 'react';
import UploadSection from "./components/UploadSection";
import LandingPage from "./components/LandingPage";
import Header from "./components/Header";
import AnalysisResults from "./components/AnalysisResults";

function App() {
  const uploadRef = useRef(null);
  const resultsRef = useRef(null);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [sliderIndex, setSliderIndex] = useState(0);

  useEffect(() => {
    if (analysisResult) {
      resultsRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [analysisResult]);

  const handleTryNowClick = () => {
    uploadRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const sendVideoToBackend = async (videoFile) => {
    const formData = new FormData();
    formData.append("video", videoFile);

    try {
      const response = await fetch("http://127.0.0.1:5000/analyze", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Failed to analyze video");
      }

      const data = await response.json();
      setAnalysisResult(data);

    } catch (error) {
      console.error("Error analyzing video:", error);
    }
  };

  const handleAnalysisComplete = (result) => {
    setAnalysisResult(result);
  };

  return (
    <>
      <Header />
      <LandingPage onTryNowClick={handleTryNowClick} />
      <div ref={uploadRef}>
        <UploadSection onAnalysisComplete={handleAnalysisComplete} onUpload={sendVideoToBackend} />
      </div>
      {analysisResult && (
        <div ref={resultsRef}>
          <AnalysisResults
            results={analysisResult}
            sliderIndex={sliderIndex}
            setSliderIndex={setSliderIndex}
          />
        </div>
      )}
    </>
  );
}

export default App;