import React, { useState, useCallback } from 'react';

interface Comment {
  id: string;
  date: Date;
  text: string;
  hatespeechDetected: boolean;
  hatespeechScore: number;
  status: 'blocked' | 'published';
  confirmed?: boolean;
}

interface PredictionResult {
  label: string;
  probability: number;
  probabilities: {
    non_hate: number;
    hate: number;
  };
}

declare global {
  interface Window {
    electronAPI: {
      predictHatespeech: (text: string) => Promise<{ success: boolean; data?: PredictionResult; error?: string }>;
    };
  }
}

function App() {
  const [commentText, setCommentText] = useState('');
  const [comments, setComments] = useState<Comment[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [lastPrediction, setLastPrediction] = useState<{ text: string; result: PredictionResult } | null>(null);
  const [apiAvailable, setApiAvailable] = useState(false);

  // Prüfe ob electronAPI verfügbar ist
  React.useEffect(() => {
    if (window.electronAPI) {
      setApiAvailable(true);
    } else {
      console.error('electronAPI nicht verfügbar');
    }
  }, []);

  const handleSubmit = useCallback(async () => {
    if (!commentText.trim() || isLoading) return;
    
    if (!window.electronAPI) {
      alert('Electron API nicht verfügbar. Bitte starten Sie die App neu.');
      return;
    }

    setIsLoading(true);
    try {
      const response = await window.electronAPI.predictHatespeech(commentText);
      
      if (response.success && response.data) {
        const hatespeechDetected = response.data.label === 'hate';
        const hatespeechScore = response.data.probabilities.hate;

        const newComment: Comment = {
          id: Date.now().toString(),
          date: new Date(),
          text: commentText,
          hatespeechDetected,
          hatespeechScore,
          status: hatespeechDetected ? 'blocked' : 'published',
        };

        setComments((prev) => [newComment, ...prev]);
        setLastPrediction({ text: commentText, result: response.data });
        setCommentText('');
      } else {
        alert(`Fehler: ${response.error || 'Unbekannter Fehler'}`);
      }
    } catch (error: any) {
      console.error('Fehler bei Vorhersage:', error);
      alert(`Fehler: ${error.message || 'Unbekannter Fehler'}`);
    } finally {
      setIsLoading(false);
    }
  }, [commentText, isLoading]);

  const handleKeyPress = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const toggleStatus = (id: string) => {
    setComments((prev) =>
      prev.map((comment) =>
        comment.id === id
          ? {
              ...comment,
              status: comment.status === 'blocked' ? 'published' : 'blocked',
            }
          : comment
      )
    );
  };

  const formatDate = (date: Date) => {
    return new Intl.DateTimeFormat('de-DE', {
      day: '2-digit',
      month: '2-digit',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    }).format(date);
  };

  const formatScore = (score: number) => {
    return (score * 100).toFixed(1) + '%';
  };

  if (!apiAvailable) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100vh', fontFamily: 'system-ui' }}>
        <div>Lade Anwendung...</div>
      </div>
    );
  }

  return (
    <div className="app-container">
      <div className="left-panel">
        <div className="comment-section">
          <h2>Kommentar eingeben</h2>
          <textarea
            className="comment-input"
            value={commentText}
            onChange={(e) => setCommentText(e.target.value)}
            onKeyDown={handleKeyPress}
            placeholder="Geben Sie hier einen Kommentar ein... (Strg+Enter zum Senden)"
            disabled={isLoading}
          />
          <button
            className="send-button"
            onClick={handleSubmit}
            disabled={isLoading || !commentText.trim()}
          >
            {isLoading ? 'Wird analysiert...' : 'Kommentar senden'}
          </button>
        </div>

        {lastPrediction && (
          <div className="prediction-result">
            <h3>Letzter gesendeter Kommentar</h3>
            <div className="prediction-text">{lastPrediction.text}</div>
            <div className="prediction-scores">
              <div className={`score-item ${lastPrediction.result.label === 'hate' ? 'hate' : 'non-hate'}`}>
                <div className="score-label">Hatespeech</div>
                <div className="score-value">{formatScore(lastPrediction.result.probabilities.hate)}</div>
              </div>
              <div className={`score-item ${lastPrediction.result.label === 'non_hate' ? 'non-hate' : 'hate'}`}>
                <div className="score-label">Kein Hatespeech</div>
                <div className="score-value">{formatScore(lastPrediction.result.probabilities.non_hate)}</div>
              </div>
            </div>
            <div style={{ marginTop: '10px', fontSize: '12px', color: '#666' }}>
              Vorhersage: <strong>{lastPrediction.result.label === 'hate' ? 'Hatespeech erkannt' : 'Kein Hatespeech'}</strong> 
              {' '}({formatScore(lastPrediction.result.probability)})
            </div>
          </div>
        )}
      </div>

      <div className="right-panel">
        <div className="table-section">
          <h2>Kommentare</h2>
          <div className="table-container">
            {comments.length === 0 ? (
              <div className="empty-table">Noch keine Kommentare vorhanden</div>
            ) : (
              <table>
                <thead>
                  <tr>
                    <th>Datum</th>
                    <th>Kommentar</th>
                    <th>Hatespeech erkannt</th>
                    <th>Status</th>
                    <th>Aktion</th>
                  </tr>
                </thead>
                <tbody>
                  {comments.map((comment) => (
                    <tr key={comment.id}>
                      <td className="date-cell">{formatDate(comment.date)}</td>
                      <td className="comment-cell">{comment.text}</td>
                      <td>
                        <span className={`hatespeech-badge ${comment.hatespeechDetected ? 'yes' : 'no'}`}>
                          {comment.hatespeechDetected ? 'Ja' : 'Nein'} ({formatScore(comment.hatespeechScore)})
                        </span>
                      </td>
                      <td>
                        <span className={`status-badge ${comment.status}`}>
                          {comment.status === 'blocked' ? 'Gesperrt' : 'Veröffentlicht'}
                        </span>
                      </td>
                      <td>
                        <div className="action-buttons">
                          <button
                            className="action-button toggle-status"
                            onClick={() => toggleStatus(comment.id)}
                            title={comment.status === 'blocked' ? 'Veröffentlichen' : 'Sperren'}
                          >
                            {comment.status === 'blocked' ? 'Veröffentlichen' : 'Sperren'}
                          </button>
                          {comment.hatespeechDetected && (
                            <>
                              <button
                                className="action-button confirm"
                                onClick={() => {
                                  // Platzhalter für zukünftige Funktion
                                  alert('Hatespeech bestätigen - Funktion noch nicht implementiert');
                                }}
                                title="Hatespeech bestätigen"
                              >
                                Bestätigen
                              </button>
                              <button
                                className="action-button reject"
                                onClick={() => {
                                  // Platzhalter für zukünftige Funktion
                                  alert('Hatespeech nicht bestätigen - Funktion noch nicht implementiert');
                                }}
                                title="Hatespeech nicht bestätigen"
                              >
                                Nicht bestätigen
                              </button>
                            </>
                          )}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;

