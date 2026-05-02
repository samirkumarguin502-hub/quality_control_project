import { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  // File select
  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    setFile(selected);
    setPreview(URL.createObjectURL(selected));
    setResult(null);
  };

  // Detect API call
  const handleDetect = async () => {
    if (!file) {
      alert("Please select a file");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      setLoading(true);

      const res = await axios.post(
        "http://127.0.0.1:8000/predict",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );

      setResult(res.data);
    } catch (error) {
      console.error(error);
      alert("Backend error ❌");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <h1>PCB AI Defect Detection</h1>

      {/* Upload */}
      <input type="file" onChange={handleFileChange} />
      <br /><br />

      {/* Preview */}
      {preview && (
        <div>
          <h3>Preview:</h3>
          <img src={preview} alt="preview" className="preview" />
        </div>
      )}

      <br />

      {/* Button */}
      <button onClick={handleDetect}>Detect</button>

      {/* Loading */}
      {loading && <p>🔄 Processing...</p>}

      {/* Results */}
      {result && (
        <div className="result-box">
          <h2>Results:</h2>

          {result.detections.length === 0 ? (
            <p>No defects found ✅</p>
          ) : (
            result.detections.map((d, i) => (
              <div key={i} className="card">
                <h3>Defect {i + 1}</h3>
                <p><b>Type:</b> {d.type}</p>
                <p>
                  <b>Confidence:</b>{" "}
                  {(d.confidence * 100).toFixed(2)}%
                </p>
              </div>
            ))
          )}
        </div>
      )}
    </div>
  );
}

export default App;