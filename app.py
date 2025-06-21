from flask import Flask, request, jsonify
import os
import poseExtraction  # 🔁 Your main analyzer logic
from flask_cors import CORS

app = Flask(
    __name__,
    static_folder="static",              # 📂 tell Flask where static files live
    static_url_path="/static"           # 🌐 make them available at /static
)
CORS(app, resources={r"/analyze": {"origins": "*"}}, supports_credentials=True)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/analyze", methods=["POST"])
def analyze():
    if "video" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["video"]
    save_path = os.path.join(UPLOAD_FOLDER, "MySwing.mp4")
    file.save(save_path)
    print('✅ Received a file')

    # Run analysis and get result with pose frame paths
    result = poseExtraction.run_analysis(save_path)

    # Return JSON to frontend
    return jsonify({
        "match": result["match"],
        "similarity": result["similarity"],
        "frames": result["frames"],           # list of user pose frame URLs
        "pro_frames": result["pro_frames"],   # list of pro pose frame URLs
        "overlay_frames": result["overlay_frames"]  # list of overlay frame URLs
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)