// import logo from "./logo.svg";
import "./App.css";
import React, { useState } from "react";
import AudioUploader from "./components/AudioUploader";
import AudioPlayer from "./components/AudioPlayer";

const App = () => {
  const [denoisedFile, setDenoisedFile] = useState(null);

  return (
    <div className="App">
      <h1>Audio Cleaning App</h1>
      <AudioUploader setDenoisedFile={setDenoisedFile} />
      <AudioPlayer file={denoisedFile} />
    </div>
  );
};

export default App;

// function App() {
//   return (
//     <div className="App">
//       <header className="App-header">
//         <img src={logo} className="App-logo" alt="logo" />
//         <p>
//           Edit <code>src/App.js</code> and save to reload.
//         </p>
//         <a
//           className="App-link"
//           href="https://reactjs.org"
//           target="_blank"
//           rel="noopener noreferrer"
//         >
//           Learn React
//         </a>
//       </header>
//     </div>
//   );
// }

// export default App;
