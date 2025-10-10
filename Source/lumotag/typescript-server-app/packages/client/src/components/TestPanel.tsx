import { useState, useEffect } from "react";
import { apiService } from "@/services/api";

export function TestPanel() {
  const [deviceMapping, setDeviceMapping] = useState<any>(null);
  const [selectedDevice, setSelectedDevice] = useState<string>("");
  const [targetTagId, setTargetTagId] = useState<string>("");
  const [tagMultiplier, setTagMultiplier] = useState<number>(1);
  const [loading, setLoading] = useState<{ [key: string]: boolean }>({});
  const [results, setResults] = useState<{ [key: string]: any }>({});

  useEffect(() => {
    apiService.getDeviceMapping().then(data => {
      setDeviceMapping(data.device_ids);
      const firstDevice = Object.keys(data.device_ids)[0];
      setSelectedDevice(firstDevice);
      setTargetTagId(data.device_ids[firstDevice].tag_id);
    });
  }, []);

  const convertImageToBase64 = async (): Promise<string> => {
    try {
      const response = await fetch("/chick.png");
      const blob = await response.blob();
      
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onloadend = () => {
          const base64 = reader.result as string;
          // Remove the "data:image/png;base64," prefix
          const base64Data = base64.split(',')[1];
          resolve(base64Data);
        };
        reader.onerror = reject;
        reader.readAsDataURL(blob);
      });
    } catch (error) {
      console.error("Failed to load chick image:", error);
      throw error;
    }
  };


  const convert2ndImageToBase64 = async (): Promise<string> => {
    try {
      const response = await fetch("/chick2.png");
      const blob = await response.blob();
      
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onloadend = () => {
          const base64 = reader.result as string;
          // Remove the "data:image/png;base64," prefix
          const base64Data = base64.split(',')[1];
          resolve(base64Data);
        };
        reader.onerror = reject;
        reader.readAsDataURL(blob);
      });
    } catch (error) {
      console.error("Failed to load chick image:", error);
      throw error;
    }
  };

  const handleTestGameState = async () => {
    const key = "gamestate";
    setResults({}); // Clear results
    setLoading({ ...loading, [key]: true });
    
    try {
      const result = await apiService.testGameState(selectedDevice);
      setResults({ [key]: result });
      console.log("GameState Result:", result);
    } catch (error) {
      console.error("GameState Error:", error);
      setResults({ [key]: { error: (error as Error).message } });
    } finally {
      setLoading({ ...loading, [key]: false });
    }
  };

  const handleTestTagAndUpload = async () => {
    const key = "tagandupload";
    setResults({}); // Clear results
    setLoading({ ...loading, [key]: true });
    
    try {
      // Generate unique image IDs
      // simulate long range and close range cameras of device
      const imageId = `test_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      const imageId2 = `test_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      
      // Convert placeholder image to base64
      const imageBase64 = await convertImageToBase64();
      const image2Base64 = await convert2ndImageToBase64();

      // Tag player (uses multiplier of 1 by default, not affected by dropdown)
      const tagResult = await apiService.testTagPlayer(selectedDevice, [targetTagId], [imageId, imageId2]);
      console.log("Tag Result:", tagResult);

      // Upload images
      const uploadResult = await apiService.testUploadImage(selectedDevice, imageId, imageBase64);
      console.log("Upload Result:", uploadResult);
      const uploadResult2 = await apiService.testUploadImage(selectedDevice, imageId2, image2Base64);
      console.log("Upload Result:", uploadResult2);
      
 
      
      setResults({ 
        [key]: { 
          imageId, imageId2,
          upload: uploadResult, 
          tag: tagResult 
        } 
      });
    } catch (error) {
      console.error("Tag and Upload Error:", error);
      setResults({ [key]: { error: (error as Error).message } });
    } finally {
      setLoading({ ...loading, [key]: false });
    }
  };


  const handleTestKillShotEvent = async () => {
    const key = "killshotevent";
    setResults({}); // Clear results
    setLoading({ ...loading, [key]: true });
    
    try {
      // Convert placeholder image to base64
      const imageBase64 = await convertImageToBase64();
      
      const result = await apiService.testKillShotEvent(
        selectedDevice, 
        deviceInfo.display_name, 
        imageBase64
      );
      setResults({ [key]: result });
      console.log("KillShot Event Result:", result);
      
      // Log image count for debugging
      console.log(`Received ${result.image_datas?.length || 0} images from ${result.display_name_tagger}`);
    } catch (error) {
      console.error("KillShot Event Error:", error);
      setResults({ [key]: { error: (error as Error).message } });
    } finally {
      setLoading({ ...loading, [key]: false });
    }
  };

  const handleReset = async () => {
    const key = "reset";
    setResults({}); // Clear results
    setLoading({ ...loading, [key]: true });
    
    try {
      const result = await apiService.resetGame();
      setResults({ [key]: result });
      console.log("Reset Result:", result);
      
      // Clear existing results after successful reset
      setTimeout(() => {
        setResults({});
      }, 3000); // Clear results after 3 seconds to show reset worked
    } catch (error) {
      console.error("Reset Error:", error);
      setResults({ [key]: { error: (error as Error).message } });
    } finally {
      setLoading({ ...loading, [key]: false });
    }
  };

  const handleTestUDPBroadcast = async () => {
    const key = "udpbroadcast";
    setResults({}); // Clear results
    setLoading({ ...loading, [key]: true });
    
    try {
      // Create array with tag_id repeated based on multiplier
      const tagIds = Array(tagMultiplier).fill(targetTagId);
      const result = await apiService.testUDPBroadcast(tagIds, []);
      setResults({ [key]: { ...result, tag_ids_sent: tagIds } });
      console.log("UDP Broadcast Result:", result);
      console.log("Tag IDs sent:", tagIds);
    } catch (error) {
      console.error("UDP Broadcast Error:", error);
      setResults({ [key]: { error: (error as Error).message } });
    } finally {
      setLoading({ ...loading, [key]: false });
    }
  };

  const handleTestTagWithUDP = async () => {
    const key = "tagwithudp";
    setResults({}); // Clear results
    setLoading({ ...loading, [key]: true });
    
    try {
      // Create array with tag_id repeated based on multiplier
      const tagIds = Array(tagMultiplier).fill(targetTagId);
      // Simulates real Python behavior - sends both HTTP and UDP with same tag_ids
      const result = await apiService.testTagPlayerWithUDP(selectedDevice, tagIds, []);
      setResults({ [key]: { ...result, tag_ids_sent: tagIds } });
      console.log("Tag with UDP Result (HTTP + UDP):", result);
      console.log("Tag IDs sent:", tagIds);
    } catch (error) {
      console.error("Tag with UDP Error:", error);
      setResults({ [key]: { error: (error as Error).message } });
    } finally {
      setLoading({ ...loading, [key]: false });
    }
  };

  if (!deviceMapping) {
    return <div className="test-panel">Loading device mapping...</div>;
  }

  const deviceInfo = deviceMapping[selectedDevice];

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
            {Object.keys(deviceMapping).map(deviceId => {
              const info = deviceMapping[deviceId];
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
            {Object.values(deviceMapping).map((info: any) => (
              <option key={info.tag_id} value={info.tag_id}>
                {info.display_name} (Tag {info.tag_id})
              </option>
            ))}
          </select>
        </div>

        <div className="control-group">
          <label>Tag ID Multiplier:</label>
          <select 
            value={tagMultiplier} 
            onChange={(e) => setTagMultiplier(Number(e.target.value))}
          >
            <option value={1}>x1 (single)</option>
            <option value={2}>x2</option>
            <option value={3}>x3</option>
            <option value={5}>x5</option>
            <option value={10}>x10</option>
          </select>
          <small>Sends tag_id repeated {tagMultiplier} time(s): [{Array(tagMultiplier).fill(targetTagId).join(', ')}]</small>
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
          onClick={handleTestKillShotEvent}
          disabled={loading.killshotevent}
          className="test-btn danger"
        >
          {loading.killshotevent ? "Processing..." : "💀 Get My KillShot Event"}
        </button>

        <button 
          onClick={handleTestUDPBroadcast}
          disabled={loading.udpbroadcast}
          className="test-btn udp"
        >
          {loading.udpbroadcast ? "Broadcasting..." : "📡 Test UDP Broadcast Only"}
        </button>

        <button 
          onClick={handleTestTagWithUDP}
          disabled={loading.tagwithudp}
          className="test-btn combo"
        >
          {loading.tagwithudp ? "Processing..." : "⚡ Tag Player (HTTP + UDP)"}
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

        .test-btn.udp {
          background: #6f42c1;
          color: white;
        }

        .test-btn.udp:hover:not(:disabled) {
          background: #5a32a3;
        }

        .test-btn.combo {
          background: #20c997;
          color: white;
        }

        .test-btn.combo:hover:not(:disabled) {
          background: #17a67e;
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
          color: #007bff;
        }
      `}</style>
    </div>
  );
}
