import React, { useState, useEffect } from 'react';
import useWebSocket, { ReadyState } from 'react-use-websocket';


const QueryComponent = () => {
  const [socketUrl] = useState('ws://localhost:8000/ws');
  const [messageHistory, setMessageHistory] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [bestResponse, setBestResponse] = useState(null);
  const [searchResults, setSearchResults] = useState([]);

  const { sendMessage, readyState } = useWebSocket(socketUrl, {
    onOpen: () => console.log('WebSocket connection established'),
    onMessage: (message) => {
      const data = JSON.parse(message.data);
      if (data.type === 'search_results') {
        setSearchResults(data.results);
      } else if (data.type === 'best_response') {
        setBestResponse(data.best_response);
      } else if (data.type === 'llm_response') {
        setMessageHistory((prev) => prev.concat(data));
      }
    },
    onError: (event) => console.error(event),
    onClose: () => console.log('WebSocket connection closed')
  });

  const handleSendMessage = () => {
    setSearchResults([]);
    setMessageHistory([]);
    setBestResponse(null);
    sendMessage(inputValue);
    setInputValue('');
  };

  const connectionStatus = {
    [ReadyState.CONNECTING]: 'Connecting',
    [ReadyState.OPEN]: 'Open',
    [ReadyState.CLOSING]: 'Closing',
    [ReadyState.CLOSED]: 'Closed',
    [ReadyState.UNINSTANTIATED]: 'Uninstantiated',
  }[readyState];

  return (
    <div className="App">
      <header className="App-header">
        <h1>LLM Response Evaluator</h1>
      </header>
      <div className="query-container">
        <div className="query-form">
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            placeholder="Enter your query"
            className="query-input"
          />
          <button onClick={handleSendMessage} disabled={readyState !== ReadyState.OPEN} className="query-button">
            Send
          </button>
        </div>
        <div className="results">
          <div className="search-results">
            <h2>Search Results:</h2>
            <ul>
              {searchResults.map((result, idx) => (
                <li key={idx}>{result}</li>
              ))}
            </ul>
          </div>
          <div className="llm-responses">
            <h2>LLM Responses:</h2>
            <ul>
              {messageHistory.map((message, idx) => (
                <li key={idx}>
                  <strong>{message.model}:</strong> {message.response}
                  <br />
                  <em>BLEU Score: {message.bleu_score}, Cosine Similarity: {message.cosine_similarity}</em>
                </li>
              ))}
            </ul>
          </div>
          {bestResponse && (
            <div className="best-response">
              <h2>Best Response:</h2>
              <p><strong>{bestResponse.model}:</strong> {bestResponse.response}</p>
              <p><em>BLEU Score: {bestResponse.bleu_score}, Cosine Similarity: {bestResponse.cosine_similarity}</em></p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default QueryComponent;
