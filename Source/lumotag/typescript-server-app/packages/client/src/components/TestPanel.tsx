import { useState } from "react";
import { apiService } from "@/services/api";

// Device mapping from device-tag-mapping.json
const DEVICE_MAPPING = {
  "abc12345": { tag_id: "3", display_name: "PretendFriend" },
  "bc1ad358bb": { tag_id: "2", display_name: "PlayerDeux" },
  "54e5b53659": { tag_id: "1", display_name: "PlayerOne" },
};

const DEVICE_IDS = Object.keys(DEVICE_MAPPING);

export function TestPanel() {
  const [selectedDevice, setSelectedDevice] = useState<string>(DEVICE_IDS[0]);
  const [targetTagId, setTargetTagId] = useState<string>("1");
  const [loading, setLoading] = useState<{ [key: string]: boolean }>({});
  const [results, setResults] = useState<{ [key: string]: any }>({});

  const convertImageToBase64 = async (): Promise<string> => {
    try {
      const response = await fetch("/placeholder.jpg");
      const blob = await response.blob();
      
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onloadend = () => {
          const base64 = reader.result as string;
          // Remove the "data:image/jpeg;base64," prefix
          const base64Data = base64.split(',')[1];
          resolve(base64Data);
        };
        reader.onerror = reject;
        reader.readAsDataURL(blob);
      });
    } catch (error) {
      console.error("Failed to load placeholder image:", error);
      throw error;
    }
  };

  const handleTestGameState = async () => {
    const key = "gamestate";
    setLoading({ ...loading, [key]: true });
    
    try {
      const result = await apiService.testGameState(selectedDevice);
      setResults({ ...results, [key]: result });
      console.log("GameState Result:", result);
    } catch (error) {
      console.error("GameState Error:", error);
      setResults({ ...results, [key]: { error: (error as Error).message } });
    } finally {
      setLoading({ ...loading, [key]: false });
    }
  };

  const handleTestTagAndUpload = async () => {
    const key = "tagandupload";
    setLoading({ ...loading, [key]: true });
    
    try {
      // Generate unique image ID
      const imageId = `test_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      
      // Convert placeholder image to base64
      const imageBase64 = await convertImageToBase64();
      
      // Upload image first
      const uploadResult = await apiService.testUploadImage(selectedDevice, imageId, imageBase64);
      console.log("Upload Result:", uploadResult);
      
      // Then tag player with same image ID
      const tagResult = await apiService.testTagPlayer(selectedDevice, targetTagId, [imageId]);
      console.log("Tag Result:", tagResult);
      
      setResults({ 
        ...results, 
        [key]: { 
          imageId,
          upload: uploadResult, 
          tag: tagResult 
        } 
      });
    } catch (error) {
      console.error("Tag and Upload Error:", error);
      setResults({ ...results, [key]: { error: (error as Error).message } });
    } finally {
      setLoading({ ...loading, [key]: false });
    }
  };

  const handleTestKillScreen = async () => {
    const key = "killscreen";
    setLoading({ ...loading, [key]: true });
    
    try {
      const result = await apiService.testKillScreen(selectedDevice);
      setResults({ ...results, [key]: result });
      console.log("KillScreen Result:", result);
      
      // Log image count for debugging
      console.log(`Received ${result.image_datas?.length || 0} images from ${result.display_name_tagger}`);
    } catch (error) {
      console.error("KillScreen Error:", error);
      setResults({ ...results, [key]: { error: (error as Error).message } });
    } finally {
      setLoading({ ...loading, [key]: false });
    }
  };

  const handleReset = async () => {
    const key = "reset";
    setLoading({ ...loading, [key]: true });
    
    try {
      const result = await apiService.resetGame();
      setResults({ ...results, [key]: result });
      console.log("Reset Result:", result);
      
      // Clear existing results after successful reset
      setTimeout(() => {
        setResults({});
      }, 3000); // Clear results after 3 seconds to show reset worked
    } catch (error) {
      console.error("Reset Error:", error);
      setResults({ ...results, [key]: { error: (error as Error).message } });
    } finally {
      setLoading({ ...loading, [key]: false });
    }
  };

  const deviceInfo = DEVICE_MAPPING[selectedDevice as keyof typeof DEVICE_MAPPING];

  return (
    <div className="test-panel">
      <h3>🧪 API Test Panel</h3>
      
      <div className="test-controls">
        <div className="control-group">
          <label>Test Device:</label>
          <select 
            value={selectedDevice} 
            onChange={(e) => setSelectedDevice(e.target.value)}
          >
            {DEVICE_IDS.map(deviceId => {
              const info = DEVICE_MAPPING[deviceId as keyof typeof DEVICE_MAPPING];
              return (
                <option key={deviceId} value={deviceId}>
                  {info.display_name} ({deviceId})
                </option>
              );
            })}
          </select>
          <small>Tag ID: {deviceInfo.tag_id}</small>
        </div>

        <div className="control-group">
          <label>Target Tag ID to Hit:</label>
          <select 
            value={targetTagId} 
            onChange={(e) => setTargetTagId(e.target.value)}
          >
            {Object.values(DEVICE_MAPPING).map(info => (
              <option key={info.tag_id} value={info.tag_id}>
                {info.display_name} (Tag {info.tag_id})
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="test-buttons">
        <button 
          onClick={handleTestGameState}
          disabled={loading.gamestate}
          className="test-btn primary"
        >
          {loading.gamestate ? "Loading..." : "🎯 Test GameState"}
        </button>

        <button 
          onClick={handleTestTagAndUpload}
          disabled={loading.tagandupload}
          className="test-btn secondary"
        >
          {loading.tagandupload ? "Processing..." : "📸 Tag Player + Upload Image"}
        </button>

        <button 
          onClick={handleTestKillScreen}
          disabled={loading.killscreen}
          className="test-btn danger"
        >
          {loading.killscreen ? "Loading..." : "💀 Test KillScreen"}
        </button>

        <button 
          onClick={handleReset}
          disabled={loading.reset}
          className="test-btn warning"
        >
          {loading.reset ? "Resetting..." : "🔄 Reset Game"}
        </button>
      </div>

      {Object.keys(results).length > 0 && (
        <div className="test-results">
          <h4>Results:</h4>
          <pre>{JSON.stringify(results, null, 2)}</pre>
        </div>
      )}

      <style>{`
        .test-panel {
          background: #f8f9fa;
          border: 1px solid #dee2e6;
          border-radius: 8px;
          padding: 1rem;
          margin: 1rem 0;
        }

        .test-controls {
          display: flex;
          gap: 1rem;
          margin-bottom: 1rem;
          flex-wrap: wrap;
        }

        .control-group {
          display: flex;
          flex-direction: column;
          gap: 0.25rem;
        }

        .control-group label {
          font-weight: 500;
          font-size: 0.9rem;
        }

        .control-group select {
          padding: 0.5rem;
          border: 1px solid #ccc;
          border-radius: 4px;
          font-size: 0.9rem;
        }

        .control-group small {
          color: #666;
          font-size: 0.8rem;
        }

        .test-buttons {
          display: flex;
          gap: 0.5rem;
          margin-bottom: 1rem;
          flex-wrap: wrap;
        }

        .test-btn {
          padding: 0.75rem 1rem;
          border: none;
          border-radius: 4px;
          cursor: pointer;
          font-weight: 500;
          transition: all 0.2s;
        }

        .test-btn:disabled {
          opacity: 0.6;
          cursor: not-allowed;
        }

        .test-btn.primary {
          background: #007bff;
          color: white;
        }

        .test-btn.primary:hover:not(:disabled) {
          background: #0056b3;
        }

        .test-btn.secondary {
          background: #28a745;
          color: white;
        }

        .test-btn.secondary:hover:not(:disabled) {
          background: #1e7e34;
        }

        .test-btn.danger {
          background: #dc3545;
          color: white;
        }

        .test-btn.danger:hover:not(:disabled) {
          background: #c82333;
        }

        .test-btn.warning {
          background: #fd7e14;
          color: white;
        }

        .test-btn.warning:hover:not(:disabled) {
          background: #e8590c;
        }

        .test-results {
          background: #fff;
          border: 1px solid #ddd;
          border-radius: 4px;
          padding: 1rem;
          max-height: 300px;
          overflow-y: auto;
        }

        .test-results h4 {
          margin-top: 0;
          margin-bottom: 0.5rem;
        }

        .test-results pre {
          font-size: 0.8rem;
          white-space: pre-wrap;
          word-break: break-word;
          margin: 0;
        }
      `}</style>
    </div>
  );
}
