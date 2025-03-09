import { useState } from 'react';
import { Music, Upload, AlertCircle } from 'lucide-react';

// Update API_URL for Vercel deployment
const API_URL = import.meta.env.VITE_API_URL;
console.log(API_URL)

const MusicGenreClassifier = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [audioUrl, setAudioUrl] = useState(null);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      if (file.type === 'audio/wav') {
        setSelectedFile(file);
        setResult(null);
        setError(null);
        setAudioUrl(URL.createObjectURL(file));
      } else {
        setError('Please select a WAV file.');
        setSelectedFile(null);
        setAudioUrl(null);
      }
    }
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (selectedFile) {
      setIsAnalyzing(true);
      setError(null);

      const formData = new FormData();
      formData.append('file', selectedFile);

      try {
        const response = await fetch(`${API_URL}/predict`, {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) {
          throw new Error(`Error: ${response.statusText}`);
        }

        const data = await response.json();
        setResult(data);
      } catch (err) {
        setError('Failed to analyze the music file. Please try again.');
        console.error('Error:', err);
      } finally {
        setIsAnalyzing(false);
      }
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <div className="max-w-2xl mx-auto">
        <div className="text-center mb-8">
          <Music className="w-16 h-16 mx-auto mb-4 text-blue-500" />
          <h1 className="text-3xl font-bold mb-2">Music Genre Classifier</h1>
          <p className="text-gray-600">Upload your WAV file to identify its genre</p>
        </div>

        <form onSubmit={handleSubmit} className="bg-white p-6 rounded-lg shadow-md">
          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Select WAV File
            </label>
            <div className="flex items-center justify-center w-full">
              <label className="w-full flex flex-col items-center px-4 py-6 bg-white rounded-lg border-2 border-dashed border-gray-300 cursor-pointer hover:border-blue-500">
                <Upload className="w-8 h-8 text-gray-400" />
                <span className="mt-2 text-gray-500">
                  {selectedFile ? selectedFile.name : 'Drop your WAV file here or click to browse'}
                </span>
                <input
                  type="file"
                  className="hidden"
                  accept=".wav"
                  onChange={handleFileChange}
                />
              </label>
            </div>
          </div>

          {audioUrl && (
            <div className="mb-6">
              <audio controls className="w-full">
                <source src={audioUrl} type="audio/wav" />
                Your browser does not support the audio element.
              </audio>
            </div>
          )}

          <button
            type="submit"
            disabled={!selectedFile || isAnalyzing}
            className="w-full py-2 px-4 border border-transparent rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:bg-gray-400 disabled:cursor-not-allowed"
          >
            {isAnalyzing ? 'Analyzing...' : 'Analyze Genre'}
          </button>
        </form>

        {error && (
          <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-md">
            <div className="flex items-center">
              <AlertCircle className="h-5 w-5 text-red-500 mr-2" />
              <p className="text-red-700">{error}</p>
            </div>
          </div>
        )}

        {result && (
          <div className="mt-6 bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4">Result</h2>
            <div className="space-y-2">
              <p className="text-gray-700">
                <span className="font-medium">Detected Genre:</span>{' '}
                <span className="text-blue-600">{result.genre}</span>
              </p>
              {/* Uncomment the following lines to display confidence */}
              {/* <p className="text-gray-700">
                <span className="font-medium">Confidence:</span>{' '}
                <span className="text-blue-600">{(result.confidence * 100).toFixed(1)}%</span>
              </p> */}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default MusicGenreClassifier;