import React from "react";

const AudioPlayer = ({ file }) => {
  return <div>{file && <audio controls src={file} />}</div>;
};

export default AudioPlayer;
