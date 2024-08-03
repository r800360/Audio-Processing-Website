import React, { useState } from "react";

const AudioUploader = () => {
  const [file, setFile] = useState(null);
  const [denoisedFile, setDenoisedFile] = useState(null);

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handleUpload = async () => {
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("https://rosachdeva.pythonanywhere.com/upload", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Network response was not ok");
      }

      const data = await response.json();
      console.log("TEST", data.denoised_file);
      setDenoisedFile(data.denoised_file);
    } catch (error) {
      console.error("Error:", error);
    }
  };

  const handleDownload = () => {
    window.open(`https://rosachdeva.pythonanywhere.com/download/${denoisedFile}`);
  };

  return (
    <div>
      <input type="file" onChange={handleFileChange} />
      <button onClick={handleUpload}>Upload and Denoise</button>
      {denoisedFile && (
        <div>
          <button onClick={handleDownload}>Download Denoised Audio</button>
          <audio controls>
            <source
              src={`https://rosachdeva.pythonanywhere.com/download/${denoisedFile}`}
              type="audio/wav"
            />
            Your browser does not support the audio element.
          </audio>
        </div>
      )}
    </div>
  );
};

export default AudioUploader;
